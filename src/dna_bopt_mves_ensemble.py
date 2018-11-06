
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
parsing.add_parse_args_mves(parser)
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
params.output_dir = params.output_dir + "_" + params.measure + "_" + params.suffix

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

    train_idx, _, test_idx = utils.train_val_test_split(indices, split=[params.init_train_examples, 0])
    val_idx = np.random.choice(test_idx, size=100, replace=False)

    train_inputs = inputs[train_idx]
    train_labels = labels[train_idx]

    top_frac_idx = [set(labels_sort_idx[-int(labels.shape[0]*per):].tolist()) for per in [0.01, 0.05, 0.1, 0.2]]

    print('label stats:', labels.mean(), labels.max(), labels.std())

    train_X = torch.FloatTensor(train_inputs).to(device)
    train_Y = torch.FloatTensor(train_labels).to(device)
    val_X = torch.FloatTensor(val_inputs).to(device)
    val_Y = torch.FloatTensor(val_labels).to(device)

    model = dbopt.get_model_nn_ensemble(inputs.shape[1], params.train_batch_size, params.num_models, params.num_hidden, sigmoid_coeff=params.sigmoid_coeff, device=params.device)
    data = [train_X, train_Y, val_X, val_Y]

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
                data,
                model,
                optim,
                unseen_reg=params.unseen_reg,
                gamma=params.gamma,
                choose_type=params.choose_type,
                normalize_fn=utils.sigmoid_standardization if params.sigmoid_coeff > 0 else utils.normal_standardization
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

            train_X_mves = train_X.clone()
            train_Y_mves = train_Y.clone()

            model_mves = copy.deepcopy(model)

            skip_idx_mves = set(train_idx)
            ack_all_mves = set()

            if os.path.exists(batch_output_dir + "/" + str(params.num_acks-1) + ".pth") and not params.clean:
                print('already done batch', ack_batch_size)
                continue

            with open(batch_output_dir + "/stats.txt", 'w' if params.clean else 'a', buffering=1) as f:
                for ack_iter in range(params.num_acks):
                    batch_ack_output_file = batch_output_dir + "/" + str(ack_iter) + ".pth"
                    if os.path.exists(batch_ack_output_file) and not params.clean:
                        checkpoint = torch.load(batch_ack_output_file)
                        model_mves.load_state_dict(checkpoint['model_state_dict'])
                        logging = checkpoint['logging']

                        model_parameters = list(model_mves.parameters())
                        optim = torch.optim.Adam(model_parameters, lr=params.retrain_lr)
                        optim = optim.load_state_dict(checkpoint['optim'])

                        ack_idx = list(checkpoint['ack_idx'].numpy())
                        ack_all_mves.update(ack_idx)
                        skip_idx_mves.update(ack_idx)
                        continue

                    test_points_idx = list(set([i for i in range(X.shape[0])]).difference(skip_idx_mves))
                    print('doing ack_iter', ack_iter, 'for mves with suffix', params.suffix)
                    model_ensemble = model_mves
                    preds, preds_vars = model_ensemble(X) # (num_candidate_points, num_samples)
                    preds = preds.detach()
                    preds_vars = preds_vars.detach()
                    #print("done predictions")

                    indices = list({i for i in range(Y.shape[0])}.difference(skip_idx_mves))
                    if params.sigmoid_coeff > 0:
                        standardized_Y = utils.sigmoid_standardization(
                                Y,
                                Y.mean(),
                                Y.std(),
                                exp=torch.exp)
                    else:
                        standardized_Y = utils.normal_standardization(
                                Y,
                                Y.mean(),
                                Y.std(),
                                exp=torch.exp)

                    log_prob_list, mse_list, kt_corr_list, std_list, mse_std_corr = bopt.get_pred_stats(
                            preds,
                            torch.sqrt(preds_vars),
                            standardized_Y, 
                            Y,
                            params.output_dist_fn, 
                            indices,
                            single_gaussian=params.single_gaussian_test_nll
                            )

                    print('log_prob_list:', log_prob_list)
                    print('mse_list:', mse_list)
                    print('kt_corr_list:', kt_corr_list)
                    print('std_list:', std_list)
                    print('mse_std_corr:', mse_std_corr)

                    skip_idx_mves.update(mves_idx)
                    ack_all_mves.update(mves_idx)
                    print('num_ack:', len(mves_idx), 'num_skip:', len(skip_idx_mves), 'num_all_ack:', len(ack_all_mves))
                    mves_idx = torch.tensor(list(mves_idx)).to(params.device)

                    new_idx = list(skip_idx_mves)
                    random.shuffle(new_idx)
                    new_idx = torch.LongTensor(new_idx)

                    ack_mves = X[mves_idx]
                    
                    train_X_mves = X.new_tensor(X[new_idx])
                    train_Y_mves = Y.new_tensor(Y[new_idx])

                    expected_num_points = (ack_iter+1)*ack_batch_size
                    assert train_X_mves.shape[0] == int(params.init_train_examples) + expected_num_points, str(train_X_mves.shape) + "[0] == " + str(int(params.init_train_examples) + expected_num_points)
                    assert train_Y_mves.shape[0] == train_X_mves.shape[0]
                    data = [train_X_mves, train_Y_mves, val_X, val_Y]
                    optim = torch.optim.Adam(list(model_mves.parameters()), lr=params.retrain_lr, weight_decay=params.retrain_l2)
                    logging, optim = dbopt.train_ensemble(
                        params, 
                        params.retrain_batch_size, 
                        params.retrain_num_epochs, 
                        data, 
                        model_mves,
                        optim,
                        unseen_reg=params.unseen_reg,
                        gamma=params.gamma,
                        choose_type=params.choose_type,
                        normalize_fn=utils.sigmoid_standardization if params.sigmoid_coeff > 0 else utils.normal_standardization
                        )

                    print(filename)
                    print('logging:', [k[-1] for k in logging])

                    f.write(str([k[-1] for k in logging]) + "\n")
                    logging = [torch.tensor(k) for k in logging]

                    ack_array = np.array(list(ack_all_mves), dtype=np.int32)

                    print('best so far:', labels[ack_array].max())

                    # inference regret computation using retrained ensemble
                    s, ir, ir_sortidx = bopt.compute_ir_regret_ensemble(
                            model_mves,
                            X,
                            labels,
                            ack_all_mves,
                            params.ack_batch_size
                        )
                    idx_frac = bopt.compute_idx_frac(ack_all_mves, top_frac_idx)
                    s += [idx_frac]
                    s = "\t".join((str(k) for k in s))

                    print(s)
                    f.write(s + "\n")

                    torch.save({
                        'model_state_dict': model_mves.state_dict(),
                        'logging': logging,
                        'optim': optim.state_dict(),
                        'ack_idx': torch.from_numpy(ack_array),
                        'ack_labels': torch.from_numpy(labels[ack_array]),
                        'best_hsic': best_hsic,
                        'diversity': len(set(sorted_preds_idx[:, -top_k:].flatten())),
                        'ir_batch': torch.from_numpy(ir),
                        'ir_batch_idx': torch.from_numpy(ir_sortidx),
                        'idx_frac': torch.tensor(idx_frac),
                        'ei_idx': torch.tensor(ei_idx),
                        'test_log_prob': torch.tensor(log_prob_list),
                        'test_mse': torch.tensor(mse_list),
                        'test_kt_corr': torch.tensor(kt_corr_list),
                        'test_std_list': torch.tensor(std_list),
                        'test_mse_std_corr': torch.tensor(mse_std_corr),
                        }, batch_ack_output_file)
                    sys.stdout.flush()
                    
            main_f.write(str(ack_batch_size) + "\t" + s + "\n")
