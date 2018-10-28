
import sys
sys.path.append('/cluster/sj1')
sys.path.append('/cluster/sj1/bb_opt/src')

import os
import torch
import random
import torch.nn as nn
from torch.nn.parameter import Parameter
from collections import namedtuple
import torch.distributions as tdist
import reparam_trainer as reparam
import numpy as np
from scipy.stats import kendalltau
import dna_bopt as dbopt
import bayesian_opt as bopt
from gpu_utils.utils import gpu_init
from tqdm import tnrange
import pandas as pd
import copy
import non_matplotlib_utils as utils

param_names = ["<num_diversity>", "<init_train_epochs>", "<init_lr>", "<retrain_epochs>", "<retrain_lr>", "<ack_batch_size>", "<mves_greedy?>", "<compare w old>", "<pred_weighting>", "<divide_by_std>", "<condense>", "<unseen reg>", "<gamma>", "<data dir>", "<filename file>", "<output dir>", "<suffix>"]

if len(sys.argv) < len(param_names)+1:
    print("Usage: python", sys.argv[0], " ".join(param_names))
    sys.exit(1)

num_diversity, init_train_epochs, train_lr, retrain_num_epochs, retrain_lr, ack_batch_size, mves_greedy, compare_w_old, pred_weighting, divide_by_std, condense, unseen_reg, gamma, data_dir, filename_file, output_dir, suffix = sys.argv[1:]

if output_dir[-1] == "/":
    output_dir = output_dir[:-1]
output_dir = output_dir + "_" + suffix

def train_val_test_split(n, split, shuffle=True):
    if type(n) == int:
        idx = np.arange(n)
    else:
        idx = n

    if shuffle:
        np.random.shuffle(idx)

    if split[0] < 1:
        assert sum(split) <= 1.
        train_end = int(n * split[0])
        val_end = train_end + int(n * split[1])
    else:
        train_end = split[0]
        val_end = train_end + split[1]

    return idx[:train_end], idx[train_end:val_end], idx[val_end:]


gpu_id = gpu_init(best_gpu_metric="mem")
print(f"Running on GPU {gpu_id}")
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

Params = namedtuple('params', [
    'output_dist_std', 
    'output_dist_fn', 
    'prior_mean',
    'prior_std',
    'num_epochs',
    'device',
    'exp_noise_samples',
    'train_lr',
    'compare_w_old',
    'pred_weighting',
    'divide_by_std',
    'condense',
    'unseen_reg',
    'gamma',
    
    'num_train_latent_samples',
    'num_latent_vars',
    'train_batch_size', 
    'hsic_train_lambda',
    
    'ack_batch_size',
    'num_acks',
    'mves_kernel_fn',
    'ack_num_model_samples',
    'ack_num_hsic_samples',
    'ack_num_pdts_points',
    'hsic_diversity_lambda',
    'mves_compute_batch_size',
    'mves_diversity',
    'mves_greedy',

    # ensemble
    'num_hidden',
    'num_models',
    'train_l2',
    'retrain_l2',
    
    'retrain_num_epochs',
    'retrain_batch_size',
    'retrain_lr',
    'hsic_retrain_lambda',
    'num_retrain_latent_samples',
])
params = Params(
    output_dist_std=1.,
    output_dist_fn=tdist.Normal, 
    exp_noise_samples=2, 
    prior_mean=0.,
    prior_std=3.,
    device='cuda',
    num_epochs=int(init_train_epochs),
    train_lr=float(train_lr),
    compare_w_old=int(compare_w_old) == 1,
    pred_weighting=int(pred_weighting),
    condense=int(condense),
    unseen_reg=unseen_reg.lower(),
    gamma=float(gamma),
    
    train_batch_size=10,
    num_latent_vars=15,
    num_train_latent_samples=20,
    hsic_train_lambda=20.,
    divide_by_std=int(divide_by_std) == 1,

    unseen_reg=unseen_reg,
    gamma=float(gamma),
    
    ack_batch_size=int(ack_batch_size),
    num_acks=20,
    mves_kernel_fn='mixrq_kernels',
    ack_num_model_samples=100,
    ack_num_hsic_samples=200,
    ack_num_pdts_points=40,
    hsic_diversity_lambda=1,
    mves_compute_batch_size=1000,
    mves_diversity=int(num_diversity),
    mves_greedy=int(mves_greedy) == 1,

    num_hidden=100,
    num_models=4,
    train_l2=0.035,
    retrain_l2=0.035,
    
    retrain_num_epochs=int(retrain_num_epochs),
    retrain_batch_size=10,
    retrain_lr=float(retrain_lr),
    hsic_retrain_lambda=20.,
    num_retrain_latent_samples=20,
)

print('compare_w_old', params.compare_w_old)
print('pred_weighting', params.pred_weighting)
print('condense', params.condense)

n_train = 20

#project = "dna_binding"
#dataset = "crx_ref_r1"

#root = "/cluster/sj1/bb_opt/"
#data_dir = root+"data/"+project+"/"+dataset+"/"

filenames = [k.strip() for k in open(filename_file).readlines()]

for filename in filenames:
    filedir = data_dir + "/" + filename + "/"
    if not os.path.exists(filedir):
        continue
    print('doing file:', filedir)
    inputs = np.load(filedir+"inputs.npy")
    labels = np.load(filedir+"labels.npy")

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    main_output_dir = output_dir + "/" + filename

    try:
        os.mkdir(main_output_dir)
    except OSError as e:
        pass

    exclude_top = 0.1

    idx = np.arange(labels.shape[0])

    labels_sort_idx = labels.argsort()
    sort_idx2 = labels_sort_idx[:-int(labels.shape[0]*exclude_top)]
    idx = idx[sort_idx2]

    train_idx, _, test_idx = utils.train_val_test_split(idx, split=[n_train, 0])
    val_idx = np.random.choice(test_idx, size=100, replace=False)

    train_inputs = inputs[train_idx]
    train_labels = labels[train_idx]

    val_inputs = inputs[val_idx]
    val_labels = labels[val_idx]

    labels_sort_idx = labels.argsort()
    top_idx = [set(labels_sort_idx[-int(labels.shape[0]*per):].tolist()) for per in [0.01, 0.05, 0.1, 0.2]]

    print('label stats:', labels.mean(), labels.max(), labels.std())

    train_label_mean = float(train_labels.mean())
    train_label_std = float(train_labels.std())

    train_labels = utils.sigmoid_standardization(train_labels, train_label_mean, train_label_std)
    val_labels = utils.sigmoid_standardization(val_labels, train_label_mean, train_label_std)

    train_X = torch.FloatTensor(train_inputs).to(device)
    train_Y = torch.FloatTensor(train_labels).to(device)
    val_X = torch.FloatTensor(val_inputs).to(device)
    val_Y = torch.FloatTensor(val_labels).to(device)

    model = dbopt.get_model_nn_ensemble(inputs.shape[1], params.train_batch_size, params.num_models, params.num_hidden, device=params.device)
    data = [train_X, train_Y, val_X, val_Y]

    init_model_path = main_output_dir + "/init_model.pth"
    loaded = False
    if os.path.isfile(init_model_path):
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
                params.num_epochs, 
                data,
                model,
                optim,
                unseen_reg=params.unseen_reg,
                gamma=params.gamma,
                )
        print('logging:', [k[-1] for k in logging])
        logging = [torch.tensor(k) for k in logging]

        torch.save({
            'model_state_dict': model.state_dict(), 
            'logging': logging,
            'optim': optim.state_dict(),
            'train_idx': torch.from_numpy(train_idx),
            'global_params': params._asdict(),
            }, init_model_path)

    X = torch.tensor(inputs, device=device)
    Y = torch.tensor(labels, device=device)

    with open(main_output_dir + "/stats.txt", 'a') as main_f:
        if not loaded:
            main_f.write(str([k[-1] for k in logging]) + "\n")

        for ack_batch_size in [5]:
            print('doing batch', ack_batch_size)
            batch_output_dir = main_output_dir + "/" + str(ack_batch_size)
            try:
                os.mkdir(batch_output_dir)
            except OSError as e:
                pass

            train_X_mves = train_X.clone()
            train_Y_mves = train_Y.clone()

            model_mves = copy.deepcopy(model)

            skip_idx_mves = set(train_idx)
            ack_all_mves = set()

            if os.path.exists(batch_output_dir + "/" + str(params.num_acks-1) + ".pth"):
                print('already done batch', ack_batch_size)
                continue

            with open(batch_output_dir + "/stats.txt", 'a', buffering=1) as f:
                for ack_iter in range(params.num_acks):
                    batch_ack_output_file = batch_output_dir + "/" + str(ack_iter) + ".pth"
                    if os.path.exists(batch_ack_output_file):
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

                    idx = list({i for i in range(Y.shape[0])}.difference(skip_idx_mves))
                    log_prob_list, mse_list = bopt.get_pred_stats(preds, Y.cpu(), train_label_mean, train_label_std, params.output_dist_fn, params.output_dist_std, idx)

                    print('log_prob_list:', log_prob_list)
                    print('mse_list:', mse_list)

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
                    if params.condense == 1:
                        print(filename, 'ei_mves_mix')
                        ei_sortidx = np.argsort(ei)
                        ei_idx = ei_sortidx[-params.mves_diversity*ack_batch_size:]
                        best_pred = torch.cat([preds[:, ei_idx], sorted_preds[:, -1].unsqueeze(-1)], dim=-1)
                    elif params.condense == 2:
                        print(filename, 'ei_condense')
                        ei_sortidx = np.argsort(ei)
                        ei_idx = ei_sortidx[-params.mves_diversity*ack_batch_size:]
                        best_pred = preds[:, ei_idx]
                        opt_weighting = torch.tensor((ei[ei_idx]-ei[ei_idx].min()))
                    elif params.condense == 3:
                        print(filename, 'ei_pdts_mix')
                        ei_sortidx = np.argsort(ei)
                        pdts_idx = bopt.get_pdts_idx(preds, params.mves_diversity*ack_batch_size, density=True)
                        print('pdts_idx:', pdts_idx)
                        ei_idx = ei_sortidx[-params.mves_diversity*ack_batch_size:]
                        best_pred = torch.cat([preds[:, ei_idx], preds[:, pdts_idx]], dim=-1)
                    elif params.condense == 4:
                        print(filename, 'cma_es')
                        idx = torch.LongTensor(list(skip_idx_mves)).to(params.device)
                        sortidx = torch.sort(Y[idx])[1]
                        idx = idx[sortidx]
                        assert idx.ndimension() == 1
                        best_pred = preds[:, idx[-10:]]
                        ei_sortidx = np.argsort(ei)
                        ei_idx = ei_sortidx[-params.mves_diversity*ack_batch_size:]
                    elif params.condense == 0:
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
                    ack_mves_vals = (Y[mves_idx]-float(train_label_mean))/float(train_label_std)
                    
                    train_X_mves = X.new_tensor(X[new_idx])
                    train_Y_mves = Y.new_tensor(Y[new_idx])

                    Y_mean = train_Y_mves.mean()
                    Y_std = train_Y_mves.std()

                    train_label_mean = float(Y_mean.item())
                    train_label_std = float(Y_std.item())

                    train_Y_mves = utils.sigmoid_standardization(train_Y_mves, Y_mean, Y_std, exp=torch.exp)
                    val_Y = utils.sigmoid_standardization(Y[val_idx], Y_mean, Y_std, exp=torch.exp)

                    expected_num_points = (ack_iter+1)*ack_batch_size
                    assert train_X_mves.shape[0] == int(n_train*0.9) + expected_num_points, str(train_X_mves.shape) + "[0] == " + str(int(n_train*0.9) + expected_num_points)
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
                        )

                    print('logging:', [k[-1] for k in logging])
                    f.write(str([k[-1] for k in logging]) + "\n")
                    logging = [torch.tensor(k) for k in logging]

                    ack_array = np.array(list(ack_all_mves), dtype=np.int32)

                    print('best so far:', labels[ack_array].max())

                    # inference regret computation using retrained ensemble
                    model_ensemble = model_mves
                    preds, _ = model_ensemble(X) # (num_candidate_points, num_samples)
                    preds = preds.transpose(0, 1)
                    ei = preds.mean(dim=0).view(-1).cpu().numpy()
                    ei_sortidx = np.argsort(ei)[-50:]
                    ack = list(ack_all_mves.union(list(ei_sortidx)))
                    best_10 = np.sort(labels[ack])[-10:]
                    best_batch_size = np.sort(labels[ack])[-ack_batch_size:]

                    idx_frac = [len(ack_all_mves.intersection(k))/len(k) for k in top_idx]

                    s = "\t".join([str(k) for k in [
                        best_10.mean(), 
                        best_10.max(), 
                        best_batch_size.mean(), 
                        best_batch_size.max(),
                        str(idx_frac)
                        ]])

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
                        'ir_batch_ei': torch.from_numpy(ei),
                        'ir_batch_ei_idx': torch.from_numpy(ei_sortidx),
                        'idx_frac': torch.tensor(idx_frac),
                        'ei_idx': torch.tensor(ei_idx),
                        #'test_log_prob': torch.tensor(log_prob_list),
                        #'test_mse': torch.tensor(mse_list),
                        }, batch_ack_output_file)
                    sys.stdout.flush()
                    
            main_f.write(str(ack_batch_size) + "\t" + s + "\n")
