
import sys
sys.path.append('/cluster/sj1')
sys.path.append('/cluster/sj1/bb_opt/src')

import os
import torch
import pprint
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
do_model_hparam_search = len(params.gammas) > 1
do_ood_val = params.ood_val > 1e-9

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

    # creates model init/data split rng
    cur_rng = ops.get_rng_state()

    np.random.seed(params.model_init_seed)
    torch.manual_seed(params.model_init_seed)
    ensemble_init_rng = ops.get_rng_state()

    np.random.seed(params.data_split_seed)
    torch.manual_seed(params.data_split_seed)
    data_split_rng = ops.get_rng_state()

    ops.set_rng_state(cur_rng)
    ###

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
    train_idx, _, test_idx, data_split_rng = utils.train_val_test_split(
            indices, 
            split=[params.init_train_examples, 0],
            rng=data_split_rng,
            )

    train_inputs = inputs[train_idx]
    train_labels = labels[train_idx]

    top_frac_idx = [set(labels_sort_idx[-int(labels.shape[0]*per):].tolist()) for per in [0.01, 0.05, 0.1, 0.2]]

    print('label stats:', labels.mean(), labels.max(), labels.std())

    train_X_init = torch.FloatTensor(train_inputs).to(device)
    train_Y_cur = torch.FloatTensor(train_labels).to(device)

    X = torch.tensor(inputs, device=device)
    Y = torch.tensor(labels, device=device)

    cur_rng = ops.get_rng_state()
    ops.set_rng_state(ensemble_init_rng)

    init_model = dbopt.get_model_nn_ensemble(
            inputs.shape[1], 
            params.train_batch_size, 
            params.num_models, 
            params.num_hidden, 
            sigmoid_coeff=params.sigmoid_coeff, 
            device=params.device
            )

    ensemble_init_rng = ops.get_rng_state()
    ops.set_rng_state(cur_rng)

    predict_info_models = None
    if params.predict_mi:
        predict_kernel_dim = 10
        predict_info_models = dbopt.PredictInfoModels(
                inputs.shape[1],
                params.ack_emb_kernel_dim,
                params.num_models,
                predict_kernel_dim,
                params.predict_stddev,
                params.predict_mmd,
                params.predict_nll,
                ).to(params.device)

    init_model_path = file_output_dir + "/init_model.pth"
    loaded = False

    preds, preds_vars = init_model(X) # (num_candidate_points, num_samples)
    preds = preds.detach()
    preds_vars = preds_vars.detach()
    bopt.get_pred_stats(
            preds,
            torch.sqrt(preds_vars),
            Y,
            train_Y_cur,
            params.output_dist_fn, 
            set(),
            params.sigmoid_coeff,
            single_gaussian=params.single_gaussian_test_nll
            )

    if os.path.isfile(init_model_path) and not params.clean:
        loaded = True
        checkpoint = torch.load(init_model_path)
        init_model.load_state_dict(checkpoint['model_state_dict'])
        logging = checkpoint["logging"]
        if "train_idx" in checkpoint:
            train_idx = checkpoint["train_idx"].numpy()
    else:

        init_model, optim, logging, best_gamma, data_split_rng = dbopt.hyper_param_train(
            params,
            init_model,
            [train_X_init, train_Y_cur, X, Y],
            data_split_rng,
            predict_info_models=predict_info_models if params.predict_mi else None,
            )

        torch.save({
            'model_state_dict': init_model.state_dict(), 
            'logging': logging,
            'best_gamma': best_gamma,
            'optim': optim.state_dict(),
            'train_idx': torch.from_numpy(train_idx),
            'global_params': vars(params),
            }, init_model_path)

    with open(file_output_dir + "/stats.txt", 'w' if params.clean else 'a', buffering=1) as main_f:
        if not loaded:
            main_f.write(pprint.pformat(logging[1]) + "\n")

        for ack_batch_size in [params.ack_batch_size]:
            print('doing batch', ack_batch_size)
            batch_output_dir = file_output_dir + "/" + str(ack_batch_size)
            try:
                os.mkdir(batch_output_dir)
            except OSError as e:
                pass

            train_X_cur = train_X_init.clone()
            train_Y_cur = train_Y_cur.clone()

            cur_model = copy.deepcopy(init_model)

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
                        cur_model.load_state_dict(checkpoint['model_state_dict'])
                        if params.predict_mi and "predict_info_models_state_dict" in checkpoint:
                            predict_info_models.load_state_dict(checkpoint["predict_info_models_state_dict"])
                        logging = checkpoint['logging']

                        model_parameters = list(cur_model.parameters())
                        optim = torch.optim.Adam(model_parameters, lr=params.retrain_lr)
                        optim = optim.load_state_dict(checkpoint['optim'])

                        ack_idx = checkpoint['ack_idx'].numpy().tolist()
                        ack_all_cur.update(ack_idx)
                        skip_idx_cur.update(ack_idx)
                        continue

                    # test stats computation
                    print('doing ack_iter', ack_iter)
                    with torch.no_grad():
                        preds, preds_vars = cur_model(X) # (num_candidate_points, num_samples)
                        preds = preds.detach()
                        preds_vars = preds_vars.detach()
                        assert preds.shape[1] == Y.shape[0], "%s[1] == %s[0]" % (str(preds.shape), str(Y.shape))

                        #preds_for_cur = torch.max(preds - train_Y_cur.max(), torch.tensor(0.).to(params.device))
                        #assert preds_for_cur.shape == preds.shape, str(preds_for_cur.shape)

                        log_prob_list, rmse_list, kt_corr_list, std_list, mse_std_corr, pred_corr = bopt.get_pred_stats(
                                preds,
                                torch.sqrt(preds_vars),
                                Y,
                                train_Y_cur,
                                params.output_dist_fn, 
                                skip_idx_cur,
                                params.sigmoid_coeff,
                                single_gaussian=params.single_gaussian_test_nll
                                )

                    print('filename:', filename, '; measure:', params.ei_diversity_measure, '; output folder', params.output_dir)

                    if "empirical_cond_" in params.ei_diversity_measure:
                        if params.ei_diversity_measure == "empirical_cond_pdts":
                            ack_to_condense = bopt.get_pdts_idx(
                                    preds, 
                                    params.num_diversity*ack_batch_size, 
                                    density=False)
                        elif params.ei_diversity_measure == "empirical_cond_er":
                            er = preds.mean(dim=0).view(-1).cpu().numpy()
                            er[:, list(skip_idx)] = er.min()
                            er_sortidx = np.argsort(er)
                            ack_to_condense = er_sortidx[-params.num_diversity*ack_batch_size:]

                        cur_ack_idx = bopt.get_empirical_condensation_ack(
                                X,
                                preds,
                                cur_model,
                                optim,
                                ack_to_condense,
                                skip_idx_cur,
                                ack_batch_size,
                                )
                    elif params.ei_diversity_measure != "info":
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

                    if params.ack_change_stat_logging:
                        rand_idx, old_std, old_nll, corr_rand, hsic_rand, mi_rand, ack_batch_hsic, rand_batch_hsic = bopt.pairwise_logging_pre_ack(
                                params,
                                preds,
                                preds_vars,
                                skip_idx_cur,
                                cur_ack_idx,
                                Y,
                                do_hsic_rand=False,
                                do_mi_rand=False,
                                do_batch_hsic=params.hsic_kernel_fn is not None,
                                )

                    ack_all_cur.update(cur_ack_idx)
                    skip_idx_cur.update(cur_ack_idx)
                    #print("cur_ack_idx:", cur_ack_idx)
                    #print('ei_labels', labels[cur_ack_idx])
                    print('ei_labels', labels[cur_ack_idx].max(), labels[cur_ack_idx].min(), labels[cur_ack_idx].mean())

                    new_idx = list(skip_idx_cur)
                    random.shuffle(new_idx)
                    new_idx = torch.LongTensor(new_idx)

                    train_X_cur = X.new_tensor(X[new_idx])
                    train_Y_cur = Y.new_tensor(Y[new_idx])

                    print("train_X_cur.shape", train_X_cur.shape)
                    
                    expected_num_points = (ack_iter+1)*ack_batch_size
                    assert train_X_cur.shape[0] == int(params.init_train_examples) + expected_num_points, str(train_X_cur.shape) + "[0] == " + str(int(params.init_train_examples) + expected_num_points)
                    assert train_Y_cur.shape[0] == train_X_cur.shape[0]

                    cur_model, ensemble_init_rng = dbopt.reinit_model(
                        params,
                        init_model,
                        cur_model,
                        ensemble_init_rng
                        )
                    cur_model, optim, logging, best_gamma, data_split_rng = dbopt.hyper_param_train(
                        params,
                        cur_model,
                        [train_X_cur, train_Y_cur, X, Y],
                        data_split_rng,
                        predict_info_models=predict_info_models if params.predict_mi else None,
                        )

                    if params.ack_change_stat_logging:
                        with torch.no_grad():
                            preds, pred_vars = cur_model(X) # (num_samples, num_points)
                            preds = preds.detach()
                            preds_vars = preds_vars.detach()

                            stats_pred = None
                            if params.predict_mi:
                                stats_pred = predict_info_models(X[rand_idx], preds[:, rand_idx], X[cur_ack_idx])

                            corr_stats = bopt.pairwise_logging_post_ack(
                                    params,
                                    preds[:, rand_idx],
                                    preds_vars[:, rand_idx],
                                    old_std,
                                    old_nll,
                                    corr_rand,
                                    Y[rand_idx],
                                    hsic_rand=hsic_rand,
                                    mi_rand=mi_rand,
                                    ack_batch_hsic=ack_batch_hsic,
                                    rand_batch_hsic=rand_batch_hsic,
                                    stats_pred=stats_pred
                                    )

                    ack_array = np.array(list(ack_all_cur), dtype=np.int32)

                    print('best so far:', labels[ack_array].max())
                    
                    # inference regret computation using retrained ensemble
                    s, ir, ir_sortidx = bopt.compute_ir_regret_ensemble(
                            cur_model,
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
                        'model_state_dict': cur_model.state_dict(), 
                        'logging': logging,
                        'optim': optim.state_dict(),
                        'best_gamma': best_gamma,
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
                        'corr_stats': torch.tensor(corr_stats) if params.ack_change_stat_logging else None,
                        }, batch_ack_output_file)
                    sys.stdout.flush()
                    
            main_f.write(str(ack_batch_size) + "\t" + s + "\n")
