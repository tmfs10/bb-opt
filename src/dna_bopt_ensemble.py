
import sys
sys.path.append('/cluster/sj1')
sys.path.append('/cluster/sj1/bb_opt/src')

import os
import torch
import random
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.distributions as tdist
import numpy as np
from scipy.stats import kendalltau, pearsonr
import dna_bopt as dbopt
import bayesian_opt as bopt
from gpu_utils.utils import gpu_init
import pandas as pd
import copy
import non_matplotlib_utils as utils
import datetime
import parsing
import argparse
import ops

parser = argparse.ArgumentParser()
parsing.add_parse_args(parser)
parsing.add_parse_args_nongrad(parser)
parsing.add_parse_args_ensemble(parser)

params = parsing.parse_args(parser)

print('PARAMS:')
for k, v in vars(params).items():
    print(k, v)

gpu_id = gpu_init(best_gpu_metric="mem")
print(f"Running on GPU {gpu_id}")
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
params.device = device

np.random.seed(params.seed)
torch.manual_seed(params.seed)

# main output dir
random_label = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
if params.output_dir[-1] == "/":
    params.output_dir = params.output_dir[:-1]
params.output_dir = params.output_dir + "_" + params.ei_diversity_measure + "_" + params.suffix

if not os.path.exists(params.output_dir):
    os.mkdir(params.output_dir)

filenames = [k.strip() for k in open(params.filename_file).readlines()][:params.num_test_tfs]

print('output_dir:', params.output_dir)

for filename in filenames:
    filedir = params.data_dir + "/" + filename + "/"
    if not os.path.exists(filedir):
        continue
    print('doing file:', filedir)
    inputs = np.load(filedir+"inputs.npy")
    labels = np.load(filedir+"labels.npy")

    if params.take_log:
        labels = np.log(labels)

    # file output dir
    file_output_dir = params.output_dir + "/" + filename
    try:
        os.mkdir(file_output_dir)
    except OSError as e:
        pass

    indices = np.arange(labels.shape[0])

    labels_sort_idx = labels.argsort()
    sort_idx = labels_sort_idx[:-int(labels.shape[0]*params.exclude_top)]
    indices = indices[sort_idx]

    # reading data
    train_idx, _, test_idx = utils.train_val_test_split(indices, split=[params.init_train_examples, 0])

    train_inputs = inputs[train_idx]
    train_labels = labels[train_idx]

    top_frac_idx = [set(labels_sort_idx[-int(labels.shape[0]*per):].tolist()) for per in [0.01, 0.05, 0.1, 0.2]]

    print('label stats:', labels.mean(), labels.max(), labels.std())

    train_X = torch.FloatTensor(train_inputs).to(device)
    train_Y = torch.FloatTensor(train_labels).to(device)

    X = torch.tensor(inputs, device=device)
    Y = torch.tensor(labels, device=device)

    # creates model
    model = dbopt.get_model_nn_ensemble(
            inputs.shape[1], 
            params.train_batch_size, 
            params.num_models, 
            params.num_hidden, 
            sigmoid_coeff=params.sigmoid_coeff, 
            device=params.device
            )
    init_model_path = file_output_dir + "/init_model.pth"
    loaded = False

    preds, preds_vars = model(X) # (num_candidate_points, num_samples)
    preds = preds.detach()
    preds_vars = preds_vars.detach()
    bopt.get_pred_stats(
            preds,
            torch.sqrt(preds_vars),
            Y,
            train_Y,
            params.output_dist_fn, 
            set(),
            params.sigmoid_coeff,
            single_gaussian=params.single_gaussian_test_nll
            )

    if os.path.isfile(init_model_path) and not params.clean:
        loaded = True
        checkpoint = torch.load(init_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        logging = checkpoint["logging"]
        if "train_idx" in checkpoint:
            train_idx = checkpoint["train_idx"].numpy()
    else:
        best_nll = float('inf')
        best_model = None
        best_optim = None
        best_logging = None
        for gamma in params.gammas:
            if len(params.gammas) > 1:
                model_copy = copy.deepcopy(model)
            else:
                model_copy = model
            optim = torch.optim.Adam(list(model_copy.parameters()), lr=params.train_lr, weight_decay=params.train_l2)
            logging, optim = dbopt.train_ensemble(
                    params, 
                    params.train_batch_size,
                    params.init_train_epochs, 
                    [train_X, train_Y],
                    model_copy,
                    optim,
                    unseen_reg=params.unseen_reg,
                    gamma=gamma,
                    choose_type=params.choose_type,
                    normalize_fn=utils.sigmoid_standardization if params.sigmoid_coeff > 0 else utils.normal_standardization,
                    early_stopping=params.early_stopping,
                    )
            val_nll_cur = logging[-1][0]
            if val_nll_cur < best_nll:
                best_nll = val_nll_cur
                best_model = model_copy
                best_optim = optim
                best_logging = logging
        logging = best_logging
        model = best_model
        optim = best_optim
        print('logging:', [k[-1] for k in logging])
        logging = [torch.tensor(k) for k in logging]

        torch.save({
            'model_state_dict': model.state_dict(), 
            'logging': logging,
            'optim': optim.state_dict(),
            'train_idx': torch.from_numpy(train_idx),
            'global_params': vars(params),
            }, init_model_path)

    with open(file_output_dir + "/stats.txt", 'w' if params.clean else 'a', buffering=1) as main_f:
        if not loaded:
            main_f.write(str([k[-1] for k in logging]) + "\n")

        for ack_batch_size in [params.ack_batch_size]:
            print('doing batch', ack_batch_size)
            batch_output_dir = file_output_dir + "/" + str(ack_batch_size)
            try:
                os.mkdir(batch_output_dir)
            except OSError as e:
                pass

            train_X_cur = train_X.clone()
            train_Y_cur = train_Y.clone()

            model_cur = copy.deepcopy(model)

            skip_idx_cur = set(train_idx.tolist())
            ack_all_cur = set()

            if os.path.exists(batch_output_dir + "/" + str(params.num_acks-1) + ".pth") and not params.clean:
                print('already done batch', ack_batch_size)
                continue

            with open(batch_output_dir + "/stats.txt", 'w' if params.clean else 'a', buffering=1) as f:
                for ack_iter in range(params.num_acks):
                    batch_ack_output_file = batch_output_dir + "/" + str(ack_iter) + ".pth"
                    if os.path.exists(batch_ack_output_file) and not params.clean:
                        checkpoint = torch.load(batch_ack_output_file)
                        model_cur.load_state_dict(checkpoint['model_state_dict'])
                        logging = checkpoint['logging']

                        model_parameters = list(model_cur.parameters())
                        optim = torch.optim.Adam(model_parameters, lr=params.retrain_lr)
                        optim = optim.load_state_dict(checkpoint['optim'])

                        ack_idx = checkpoint['ack_idx'].numpy().tolist()
                        ack_all_cur.update(ack_idx)
                        skip_idx_cur.update(ack_idx)
                        continue

                    # test stats computation
                    print('doing ack_iter', ack_iter)
                    with torch.no_grad():
                        model_ensemble = model_cur
                        preds, preds_vars = model_ensemble(X) # (num_candidate_points, num_samples)
                        preds = preds.detach()
                        preds_vars = preds_vars.detach()
                        assert preds.shape[1] == Y.shape[0], "%s[1] == %s[0]" % (str(preds.shape), str(Y.shape))

                        #preds_for_cur = torch.max(preds - train_Y_cur.max(), torch.tensor(0.).to(params.device))
                        #assert preds_for_cur.shape == preds.shape, str(preds_for_cur.shape)

                        log_prob_list, rmse_list, kt_corr_list, std_list, mse_std_corr, pred_corr = bopt.get_pred_stats(
                                preds,
                                torch.sqrt(preds_vars),
                                Y,
                                train_Y,
                                params.output_dist_fn, 
                                skip_idx_cur,
                                params.sigmoid_coeff,
                                single_gaussian=params.single_gaussian_test_nll
                                )

                    print('filename:', filename, '; measure:', params.ei_diversity_measure, '; output folder', params.output_dir)

                    if params.ei_diversity_measure != "info":
                        cur_ack_idx = bopt.get_noninfo_ack(
                                params,
                                params.ei_diversity_measure,
                                preds,
                                ack_batch_size,
                                skip_idx_cur,
                                )
                    else:
                        cur_ack_idx = bopt.get_info_ack(
                                params,
                                preds,
                                ack_batch_size,
                                skip_idx_cur,
                                )

                    with torch.no_grad():
                        unseen_idx_rand = list(set(range(Y.shape[0])).difference(skip_idx_cur.union(cur_ack_idx)))
                        rand_idx = np.random.choice(unseen_idx_rand, 100, replace=False).tolist()
                        corr_rand = np.corrcoef(preds[:, rand_idx].transpose(0, 1).detach().cpu().numpy())
                        hsic_rand = bopt.pairwise_hsic(params, preds[:, rand_idx])
                        old_std = preds.std(dim=0)

                    ack_all_cur.update(cur_ack_idx)
                    skip_idx_cur.update(cur_ack_idx)
                    print("cur_ack_idx:", cur_ack_idx)
                    print('ei_labels', labels[cur_ack_idx])

                    new_idx = list(skip_idx_cur)
                    random.shuffle(new_idx)
                    new_idx = torch.LongTensor(new_idx)

                    train_X_cur = X.new_tensor(X[new_idx])
                    train_Y_cur = Y.new_tensor(Y[new_idx])

                    print("train_X_cur.shape", train_X_cur.shape)
                    
                    expected_num_points = (ack_iter+1)*ack_batch_size
                    assert train_X_cur.shape[0] == int(params.init_train_examples) + expected_num_points, str(train_X_cur.shape) + "[0] == " + str(int(params.init_train_examples) + expected_num_points)
                    assert train_Y_cur.shape[0] == train_X_cur.shape[0]
                    best_nll = float('inf')
                    best_model = None
                    best_optim = None
                    best_logging = None
                    for gamma in params.gammas:
                        if len(params.gammas) > 1:
                            model_copy = copy.deepcopy(model)
                        else:
                            model_copy = model
                        optim = torch.optim.Adam(list(model_copy.parameters()), lr=params.retrain_lr, weight_decay=params.retrain_l2)
                        logging, optim = dbopt.train_ensemble(
                            params,
                            params.retrain_batch_size,
                            params.retrain_num_epochs,
                            [train_X_cur, train_Y_cur], 
                            model_copy,
                            optim,
                            unseen_reg=params.unseen_reg,
                            gamma=gamma,
                            choose_type=params.choose_type,
                            normalize_fn=utils.sigmoid_standardization if params.sigmoid_coeff > 0 else utils.normal_standardization,
                            early_stopping=params.early_stopping,
                            )
                        val_nll_cur = logging[-1][0]
                        if val_nll_cur < best_nll:
                            best_nll = val_nll_cur
                            best_model = model_copy
                            best_optim = optim
                            best_logging = logging
                    logging = best_logging
                    model_cur = best_model
                    optim = best_optim

                    print(filename)
                    print('logging:', [k[-1] for k in logging])

                    f.write(str([k[-1] for k in logging]) + "\n")
                    logging = [torch.tensor(k) for k in logging]

                    with torch.no_grad():
                        preds, _ = model_cur(X) # (num_samples, num_points)
                        preds = preds.detach()
                        corr_rand = corr_rand[np.tril_indices(corr_rand.shape[0], k=-1)]
                        hsic_rand = hsic_rand[np.tril_indices(hsic_rand.shape[0], k=-1)]
                        std_ratio_matrix = bopt.compute_std_ratio_matrix(old_std[rand_idx], preds[:, rand_idx])

                        p1 = pearsonr(corr_rand, std_ratio_matrix)[0]
                        kt = kendalltau(corr_rand, std_ratio_matrix)[0]
                        print('corr pearson:', p1, ';', 'kt:', kt)
                        p1 = pearsonr(hsic_rand, std_ratio_matrix)[0]
                        kt = kendalltau(hsic_rand, std_ratio_matrix)[0]
                        print('hsic pearson:', p1, ';', 'kt:', kt)

                    ack_array = np.array(list(ack_all_cur), dtype=np.int32)

                    print('best so far:', labels[ack_array].max())
                    
                    # inference regret computation using retrained ensemble
                    s, ir, ir_sortidx = bopt.compute_ir_regret_ensemble(
                            model_cur,
                            X,
                            labels,
                            ack_all_cur,
                            params.ack_batch_size
                        )
                    idx_frac = bopt.compute_idx_frac(ack_all_cur, top_frac_idx)
                    s += [idx_frac]
                    s = "\t".join((str(k) for k in s))

                    print(s)
                    f.write(s + "\n")

                    torch.save({
                        'model_state_dict': model_cur.state_dict(), 
                        'logging': logging,
                        'optim': optim.state_dict(),
                        'ack_idx': torch.from_numpy(ack_array),
                        'ack_labels': torch.from_numpy(labels[ack_array]),
                        'ir_batch_cur': torch.from_numpy(ir),
                        'ir_batch_cur_idx': torch.from_numpy(ir_sortidx),
                        'idx_frac': torch.tensor(idx_frac),
                        'test_log_prob': torch.tensor(log_prob_list),
                        'test_mse': torch.tensor(rmse_list),
                        'test_kt_corr': torch.tensor(kt_corr_list),
                        'test_std_list': torch.tensor(std_list),
                        'test_mse_std_corr': torch.tensor(mse_std_corr),
                        'test_pred_corr': torch.tensor(pred_corr),
                        }, batch_ack_output_file)
                    sys.stdout.flush()
                    
            main_f.write(str(ack_batch_size) + "\t" + s + "\n")
