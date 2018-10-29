
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

    val_inputs = inputs[val_idx]
    val_labels = labels[val_idx]

    top_frac_idx = [set(labels_sort_idx[-int(labels.shape[0]*per):].tolist()) for per in [0.01, 0.05, 0.1, 0.2]]

    print('label stats:', labels.mean(), labels.max(), labels.std())

    Y_mean = float(train_labels.mean())
    Y_std = float(train_labels.std())

    train_labels = utils.sigmoid_standardization(train_labels, Y_mean, Y_std)
    val_labels = utils.sigmoid_standardization(val_labels, Y_mean, Y_std)

    train_X = torch.FloatTensor(train_inputs).to(device)
    train_Y = torch.FloatTensor(train_labels).to(device)
    val_X = torch.FloatTensor(val_inputs).to(device)
    val_Y = torch.FloatTensor(val_labels).to(device)

    model = dbopt.get_model_nn_ensemble(inputs.shape[1], params.train_batch_size, params.num_models, params.num_hidden, device=params.device)
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

    with open(file_output_dir + "/stats.txt", 'a') as main_f:
        if not loaded:
            main_f.write(str([k[-1] for k in logging]) + "\n")

        for ack_batch_size in [5]:
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

            with open(batch_output_dir + "/stats.txt", 'a', buffering=1) as f:
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
                    print('doing ack_iter', ack_iter, 'for mves with suffix', suffix)
                    model_ensemble = model_mves
                    preds, _ = model_ensemble(X) # (num_candidate_points, num_samples)
                    preds = preds.transpose(0, 1)
                    #print("done predictions")

                    indices = list({i for i in range(Y.shape[0])}.difference(skip_idx_mves))
                    standardized_Y = utils.sigmoid_standardization(
                            Y,
                            train_Y_ei.mean(),
                            train_Y_ei.std(),
                            exp=torch.exp)

                    log_prob_list, mse_list, kt_corr_list = bopt.get_pred_stats(
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

                    if not params.compare_w_old:
                        preds[:, list(skip_idx_mves)] = preds.min()
                        ei = preds.mean(dim=0).view(-1).cpu().numpy()
                        std = preds.std(dim=0).view(-1).cpu().numpy()
                    else:
                        preds2 = preds.clone()
                        preds2[:, list(skip_idx_mves)] = preds2.min()
                        ei = preds2.mean(dim=0).view(-1).cpu().numpy()
                        std = preds2.std(dim=0).view(-1).cpu().numpy()
                    
                    top_k = params.mves_diversity
                    
                    sorted_preds_idx = bopt.argsort_preds(preds)
                    f.write('diversity\t' + str(len(set(sorted_preds_idx[:, -top_k:].flatten()))) + "\n")
                    sorted_preds = torch.sort(preds, dim=1)[0]    
                    #best_pred = preds.max(dim=1)[0].view(-1)
                    #best_pred = sorted_preds[:, -top_k:]

                    opt_weighting = None
                    if params.measure == 'ei_mves_mix':
                        print(filename, 'ei_mves_mix')
                        ei_sortidx = np.argsort(ei)
                        ei_idx = ei_sortidx[-params.mves_diversity*ack_batch_size:]
                        best_pred = torch.cat([preds[:, ei_idx], sorted_preds[:, -1].unsqueeze(-1)], dim=-1)
                    elif params.measure == 'ei_condense':
                        print(filename, 'ei_condense')
                        ei_sortidx = np.argsort(ei)
                        ei_idx = ei_sortidx[-params.mves_diversity*ack_batch_size:]
                        best_pred = preds[:, ei_idx]
                        opt_weighting = torch.tensor((ei[ei_idx]-ei[ei_idx].min()))
                    elif params.measure == 'ei_pdts_mix':
                        print(filename, 'ei_pdts_mix')
                        ei_sortidx = np.argsort(ei)
                        pdts_idx = bopt.get_pdts_idx(preds, params.mves_diversity*ack_batch_size, density=True)
                        print('pdts_idx:', pdts_idx)
                        ei_idx = ei_sortidx[-params.mves_diversity*ack_batch_size:]
                        best_pred = torch.cat([preds[:, ei_idx], preds[:, pdts_idx]], dim=-1)
                    elif params.measure == 'cma_es':
                        print(filename, 'cma_es')
                        indices = torch.LongTensor(list(skip_idx_mves)).to(params.device)
                        sortidx = torch.sort(Y[indices])[1]
                        indices = indices[sortidx]
                        assert indices.ndimension() == 1
                        best_pred = preds[:, indices[-10:]]
                        ei_sortidx = np.argsort(ei)
                        ei_idx = ei_sortidx[-params.mves_diversity*ack_batch_size:]
                    elif params.measure == 'mves':
                        print(filename, 'mves')
                        ei_sortidx = np.argsort(ei)
                        ei_idx = ei_sortidx[-params.mves_diversity*ack_batch_size:]
                        best_pred = sorted_preds[:, -top_k:]
                    else:
                        assert False

                    print('best_pred.shape\t' + str(best_pred.shape))
                    f.write('best_pred.shape\t' + str(best_pred.shape))

                    print('best_pred:', best_pred.mean(0), best_pred.std(0))

                    #best_pred = (best_pred - best_pred.mean(0))/best_pred.std()
                    
                    mves_compute_batch_size = params.mves_compute_batch_size
                    #ack_batch_size=params.ack_batch_size
                    mves_idx, best_hsic = bopt.acquire_batch_mves_sid(
                            params,
                            best_pred, 
                            preds, 
                            skip_idx_mves, 
                            mves_compute_batch_size, 
                            ack_batch_size, 
                            true_labels=labels, 
                            greedy_ordering=params.mves_greedy, 
                            pred_weighting=params.pred_weighting, 
                            normalize=True, 
                            divide_by_std=params.divide_by_std, 
                            opt_weighting=None)

                    print('ei_idx', ei_idx)
                    print('mves_idx', mves_idx)
                    print('intersection size', len(set(mves_idx).intersection(set(ei_idx.tolist()))))

                    if len(mves_idx) < ack_batch_size:
                        for idx in ei_idx[::-1]:
                            idx = int(idx)
                            if len(mves_idx) >= ack_batch_size:
                                break
                            if idx in mves_idx and idx not in skip_idx_mves:
                                continue
                            mves_idx += [idx]
                    assert len(mves_idx) == ack_batch_size
                    assert len(set(mves_idx)) == ack_batch_size

                    print('ei_labels', labels[ei_idx])
                    print('mves_labels', labels[list(mves_idx)])

                    print('best_hsic\t' + str(best_hsic))
                    print("train_X.shape:", train_X_mves.shape)

                    f.write('best_hsic\t' + str(best_hsic) + "\n")
                    f.write('train_X.shape\t' + str(train_X_mves.shape) + "\n")

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

                    Y_mean = train_Y_mves.mean()
                    Y_std = train_Y_mves.std()

                    train_Y_mves = utils.sigmoid_standardization(
                            train_Y_mves, 
                            Y_mean, 
                            Y_std, 
                            exp=torch.exp)
                    val_Y = utils.sigmoid_standardization(Y[val_idx], Y_mean, Y_std, exp=torch.exp)

                    expected_num_points = (ack_iter+1)*ack_batch_size
                    assert train_X_mves.shape[0] == int(params.init_train_examples) + expected_num_points, str(train_X_mves.shape) + "[0] == " + str(int(params.init_train_examples) + expected_num_points)
                    assert train_Y_mves.shape[0] == train_X_mves.shape[0]
                    data = [train_X_mves, train_Y_mves, val_X, val_Y]
                    optim = torch.optim.Adam(list(model_ei.parameters()), lr=params.retrain_lr, weight_decay=params.retrain_l2)
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
                        )

                    print(filename)
                    print('logging:', [k[-1] for k in logging])

                    f.write(str([k[-1] for k in logging]) + "\n")
                    logging = [torch.tensor(k) for k in logging]

                    ack_array = np.array(list(ack_all_mves), dtype=np.int32)

                    print('best so far:', labels[ack_array].max())

                    # inference regret computation using retrained ensemble
                    s, idx_frac, ir_ei, ir_ei_sortidx = bopt.compute_ir_regret_ensemble(
                            model_mves,
                            X,
                            labels,
                            ack_all_mves,
                            top_frac_idx,
                            params.ack_batch_size
                        )

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
                        'ir_batch_ei': torch.from_numpy(ir_ei),
                        'ir_batch_ei_idx': torch.from_numpy(ir_ei_sortidx),
                        'idx_frac': torch.tensor(idx_frac),
                        'ei_idx': torch.tensor(ei_idx),
                        'test_log_prob': torch.tensor(log_prob_list),
                        'test_mse': torch.tensor(mse_list),
                        'test_kt_corr': torch.tensor(kt_corr_list),
                        }, batch_ack_output_file)
                    sys.stdout.flush()
                    
            main_f.write(str(ack_batch_size) + "\t" + s + "\n")
