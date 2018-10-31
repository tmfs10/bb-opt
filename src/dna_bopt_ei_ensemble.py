
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
from scipy.stats import kendalltau
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
parsing.add_parse_args_ei(parser)
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

    if os.path.isfile(init_model_path) and not params.clean:
        loaded = True
        checkpoint = torch.load(init_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        logging = checkpoint["logging"]
        if "train_idx" in checkpoint:
            train_idx = checkpoint["train_idx"].numpy()
    else:
        optim = torch.optim.Adam(list(model.parameters()), lr=params.train_lr, weight_decay=params.train_l2)
        logging, optim = dbopt.train_ensemble(
                params, 
                params.train_batch_size,
                params.init_train_epochs, 
                [train_X, train_Y],
                model,
                optim,
                unseen_reg=params.unseen_reg,
                gamma=params.gamma,
                choose_type=params.choose_type,
                normalize_fn=utils.sigmoid_standardization if params.sigmoid_coeff > 0 else utils.normal_standardization,
                early_stopping=params.early_stopping,
                )
        print('logging:', [k[-1] for k in logging])
        logging = [torch.tensor(k) for k in logging]

        torch.save({
            'model_state_dict': model.state_dict(), 
            'logging': logging,
            'optim': optim.state_dict(),
            'train_idx': torch.from_numpy(train_idx),
            'global_params': vars(params),
            }, init_model_path)

    X = torch.tensor(inputs, device=device)
    Y = torch.tensor(labels, device=device)

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

            train_X_ei = train_X.clone()
            train_Y_ei = train_Y.clone()

            model_ei = copy.deepcopy(model)

            skip_idx_ei = set(train_idx)
            ack_all_ei = set()

            if os.path.exists(batch_output_dir + "/" + str(params.num_acks-1) + ".pth") and not params.clean:
                print('already done batch', ack_batch_size)
                continue

            with open(batch_output_dir + "/stats.txt", 'w' if params.clean else 'a', buffering=1) as f:
                for ack_iter in range(params.num_acks):
                    batch_ack_output_file = batch_output_dir + "/" + str(ack_iter) + ".pth"
                    if os.path.exists(batch_ack_output_file) and not params.clean:
                        checkpoint = torch.load(batch_ack_output_file)
                        model_ei.load_state_dict(checkpoint['model_state_dict'])
                        logging = checkpoint['logging']

                        model_parameters = list(model_ei.parameters())
                        optim = torch.optim.Adam(model_parameters, lr=params.retrain_lr)
                        optim = optim.load_state_dict(checkpoint['optim'])

                        ack_idx = list(checkpoint['ack_idx'].numpy())
                        ack_all_ei.update(ack_idx)
                        skip_idx_ei.update(ack_idx)
                        continue

                    # test stats computation
                    print('doing ack_iter', ack_iter)
                    with torch.no_grad():
                        model_ensemble = model_ei
                        preds, preds_vars = model_ensemble(X) # (num_candidate_points, num_samples)
                        preds = preds.detach()
                        preds_vars = preds_vars.detach()
                        assert preds.shape[1] == Y.shape[0], "%s[1] == %s[0]" % (str(preds.shape), str(Y.shape))

                        #preds_for_ei = torch.max(preds - train_Y_ei.max(), torch.tensor(0.).to(params.device))
                        #assert preds_for_ei.shape == preds.shape, str(preds_for_ei.shape)

                        indices = list({i for i in range(Y.shape[0])}.difference(skip_idx_ei))
                        if params.sigmoid_coeff > 0:
                            standardized_Y = utils.sigmoid_standardization(
                                    Y,
                                    train_Y.mean(),
                                    train_Y.std(),
                                    exp=torch.exp)
                        else:
                            standardized_Y = utils.normal_standardization(
                                    Y,
                                    train_Y.mean(),
                                    train_Y.std(),
                                    exp=torch.exp)

                        log_prob_list, rmse_list, kt_corr_list, std_list, mse_std_corr = bopt.get_pred_stats(
                                preds,
                                torch.sqrt(preds_vars),
                                standardized_Y,
                                Y,
                                params.output_dist_fn, 
                                indices,
                                single_gaussian=params.single_gaussian_test_nll
                                )

                        print('log_prob_list:', log_prob_list)
                        print('rmse_list:', rmse_list)
                        print('kt_corr_list:', kt_corr_list)
                        print('std_list:', std_list)
                        print('mse_std_corr:', mse_std_corr)

                        ei = preds.mean(dim=0).view(-1).cpu().numpy()
                        std = preds.std(dim=0).view(-1).cpu().numpy()

                        print('filename:', filename, '; measure:', params.ei_diversity_measure, '; output folder', params.output_dir)
                        if "var" in params.ei_diversity_measure:
                            ei_sortidx = np.argsort(ei/std)
                        elif "ucb" in params.ei_diversity_measure:
                            ei_sortidx = np.argsort(ei + params.ucb*std)

                    if "none" in params.ei_diversity_measure:
                        ei_idx = []
                        for idx in ei_sortidx[::-1]:
                            if idx not in skip_idx_ei:
                                ei_idx += [idx]
                            if len(ei_idx) >= ack_batch_size:
                                break
                    elif "hsic" in params.ei_diversity_measure:
                        ei_idx = bopt.ei_diversity_selection_hsic(
                                params, 
                                preds, 
                                skip_idx_ei, 
                                device=params.device)
                    elif "detk" in params.ei_diversity_measure:
                        ei_idx = bopt.ei_diversity_selection_detk(
                                params, 
                                preds, 
                                skip_idx_ei, 
                                device=params.device)
                    elif "pdts" in params.ei_diversity_measure:
                        ei_idx = set()
                        ei_sortidx = np.argsort(ei)
                        sorted_preds_idx = []
                        for i in range(preds.shape[0]):
                            sorted_preds_idx += [np.argsort(preds[i].numpy())]
                        sorted_preds_idx = np.array(sorted_preds_idx)
                        if "density" in params.ei_diversity_measure:
                            counts = np.zeros(sorted_preds_idx.shape[1])
                            for rank in range(sorted_preds_idx.shape[1]):
                                counts[:] = 0
                                for idx in sorted_preds_idx[:, rank]:
                                    counts[idx] += 1
                                counts_idx = counts.argsort()[::-1]
                                j = 0
                                while len(ei_idx) < ack_batch_size and j < counts_idx.shape[0] and counts[counts_idx[j]] > 0:
                                    idx = int(counts_idx[j])
                                    ei_idx.update({idx})
                                    j += 1
                                if len(ei_idx) >= ack_batch_size:
                                    break
                        else:
                            assert params.ei_diversity_measure == "pdts", params.ei_diversity_measure
                            for i_model in range(sorted_preds_idx.shape[0]):
                                for idx in sorted_preds_idx[i_model]:
                                    idx2 = int(idx)
                                    if idx2 not in ei_idx:
                                        ei_idx.update({idx2})
                                        break
                                if len(ei_idx) >= ack_batch_size:
                                    break
                        ei_idx = list(ei_idx)
                    else:
                        assert False, "Not implemented " + params.ei_diversity_measure
                    assert len(ei_idx) == ack_batch_size, len(ei_idx)

                    best_ei_10 = labels[ei_sortidx[-10:]]
                    f.write('best_ei_10\t' + str(best_ei_10.mean()) + "\t" + str(best_ei_10.max()) + "\t")

                    ack_all_ei.update(ei_idx)
                    skip_idx_ei.update(ei_idx)
                    print("ei_idx:", ei_idx)
                    print('ei_labels', labels[ei_idx])

                    new_idx = list(skip_idx_ei)
                    random.shuffle(new_idx)
                    new_idx = torch.LongTensor(new_idx)

                    train_X_ei = X.new_tensor(X[new_idx])
                    train_Y_ei = Y.new_tensor(Y[new_idx])

                    print("train_X_ei.shape", train_X_ei.shape)
                    
                    expected_num_points = (ack_iter+1)*ack_batch_size
                    assert train_X_ei.shape[0] == int(params.init_train_examples) + expected_num_points, str(train_X_ei.shape) + "[0] == " + str(int(params.init_train_examples) + expected_num_points)
                    assert train_Y_ei.shape[0] == train_X_ei.shape[0]
                    optim = torch.optim.Adam(list(model_ei.parameters()), lr=params.retrain_lr, weight_decay=params.retrain_l2)
                    logging, optim = dbopt.train_ensemble(
                        params, 
                        params.retrain_batch_size, 
                        params.retrain_num_epochs, 
                        [train_X_ei, train_Y_ei], 
                        model_ei,
                        optim,
                        unseen_reg=params.unseen_reg,
                        gamma=params.gamma,
                        choose_type=params.choose_type,
                        normalize_fn=utils.sigmoid_standardization if params.sigmoid_coeff > 0 else utils.normal_standardization,
                        early_stopping=params.early_stopping,
                        )

                    print(filename)
                    print('logging:', [k[-1] for k in logging])

                    f.write(str([k[-1] for k in logging]) + "\n")
                    logging = [torch.tensor(k) for k in logging]

                    ack_array = np.array(list(ack_all_ei), dtype=np.int32)

                    print('best so far:', labels[ack_array].max())
                    
                    # inference regret computation using retrained ensemble
                    s, ir, ir_sortidx = bopt.compute_ir_regret_ensemble(
                            model_ei,
                            X,
                            labels,
                            ack_all_ei,
                            params.ack_batch_size
                        )
                    idx_frac = bopt.compute_idx_frac(ack_all_ei, top_frac_idx)
                    s += [idx_frac]
                    s = "\t".join((str(k) for k in s))

                    print(s)
                    f.write(s + "\n")

                    torch.save({
                        'model_state_dict': model_ei.state_dict(), 
                        'logging': logging,
                        'optim': optim.state_dict(),
                        'ack_idx': torch.from_numpy(ack_array),
                        'ack_labels': torch.from_numpy(labels[ack_array]),
                        'ir_batch_ei': torch.from_numpy(ir),
                        'ir_batch_ei_idx': torch.from_numpy(ir_sortidx),
                        'idx_frac': torch.tensor(idx_frac),
                        'test_log_prob': torch.tensor(log_prob_list),
                        'test_mse': torch.tensor(rmse_list),
                        'test_kt_corr': torch.tensor(kt_corr_list),
                        'test_std_list': torch.tensor(std_list),
                        'test_mse_std_corr': torch.tensor(mse_std_corr),
                        }, batch_ack_output_file)
                    sys.stdout.flush()
                    
            main_f.write(str(ack_batch_size) + "\t" + s + "\n")
