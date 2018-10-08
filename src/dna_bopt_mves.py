
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

if len(sys.argv) < 15:
    print("Usage: python", sys.argv[0], "<num_diversity> <init_train_epochs> <init_lr> <retrain_epochs> <retrain_lr> <ack_batch_size> <mves_greedy?> <compare w old> <pred_weighting> <divide_by_std> <condense> <data dir> <filename file> <output dir> <suffix>")
    sys.exit(1)

num_diversity, init_train_epochs, train_lr, retrain_num_epochs, retrain_lr, ack_batch_size, mves_greedy, compare_w_old, pred_weighting, divide_by_std, condense, data_dir, filename_file, output_dir, suffix = sys.argv[1:]

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
    
    train_batch_size=10,
    num_latent_vars=15,
    num_train_latent_samples=20,
    hsic_train_lambda=20.,
    divide_by_std=int(divide_by_std) == 1,
    
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

    train_idx, _, _ = train_val_test_split(idx, split=[n_train, 0])
    train_idx2, _, test_idx2 = train_val_test_split(n_train, split=[0.9, 0])

    test_idx = train_idx[test_idx2]
    train_idx = train_idx[train_idx2]

    train_inputs = inputs[train_idx]
    train_labels = labels[train_idx]

    val_inputs = inputs[test_idx]
    val_labels = labels[test_idx]

    labels_sort_idx = labels.argsort()
    top_idx = [set(labels_sort_idx[-int(labels.shape[0]*per):].tolist()) for per in [0.01, 0.05, 0.1, 0.2]]

    print('label stats:', labels.mean(), labels.max(), labels.std())

    train_label_mean = train_labels.mean()
    train_label_std = train_labels.std()

    train_labels = (train_labels - train_label_mean) / train_label_std
    val_labels = (val_labels - train_label_mean) / train_label_std

    train_X = torch.FloatTensor(train_inputs).to(device)
    train_Y = torch.FloatTensor(train_labels).to(device)
    val_X = torch.FloatTensor(val_inputs).to(device)
    val_Y = torch.FloatTensor(val_labels).to(device)

    model, qz, e_dist = dbopt.get_model_nn(params.prior_mean, params.prior_std, inputs.shape[1], params.num_latent_vars, device=params.device)
    data = [train_X, train_Y, val_X, val_Y]

    init_model_path = main_output_dir + "/init_model.pth"
    loaded = False
    if os.path.isfile(init_model_path):
        loaded = True
        checkpoint = torch.load(init_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        qz.load_state_dict(checkpoint['qz_state_dict'])
        logging = checkpoint["logging"]
        if "train_idx" in checkpoint:
            train_idx = checkpoint["train_idx"].numpy()
    else:
        logging, optim = dbopt.train(params, params.train_batch_size, params.train_lr, params.num_epochs, params.hsic_train_lambda, params.num_train_latent_samples, data, model, qz, e_dist)

        print('logging:', [k[-1] for k in logging])
        logging = [torch.tensor(k) for k in logging]

        torch.save({
            'model_state_dict': model.state_dict(), 
            'qz_state_dict': qz.state_dict(),
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
            qz_mves = copy.deepcopy(qz)

            skip_idx_mves = set(train_idx)
            ack_all_mves = set()
            e = reparam.generate_prior_samples(params.ack_num_model_samples, e_dist)

            if os.path.exists(batch_output_dir + "/" + str(params.num_acks-1) + ".pth"):
                print('alread done batch', ack_batch_size)
                continue

            with open(batch_output_dir + "/stats.txt", 'a', buffering=1) as f:
                for ack_iter in range(params.num_acks):
                    batch_ack_output_file = batch_output_dir + "/" + str(ack_iter) + ".pth"
                    if os.path.exists(batch_ack_output_file):
                        checkpoint = torch.load(batch_ack_output_file)
                        model_mves.load_state_dict(checkpoint['model_state_dict'])
                        qz_mves.load_state_dict(checkpoint['qz_state_dict'])
                        logging = checkpoint['logging']

                        model_parameters = list(model_mves.parameters()) + list(qz_mves.parameters())
                        optim = torch.optim.Adam(model_parameters, lr=params.retrain_lr)
                        optim = optim.load_state_dict(checkpoint['optim'])

                        ack_idx = list(checkpoint['ack_idx'].numpy())
                        ack_all_mves.update(ack_idx)
                        skip_idx_mves.update(ack_idx)
                        continue

                    print('doing ack_iter', ack_iter, 'for mves with suffix', suffix)
                    e = reparam.generate_prior_samples(params.ack_num_hsic_samples, e_dist)
                    model_ensemble = reparam.generate_ensemble_from_stochastic_net(model_mves, qz_mves, e)
                    preds = model_ensemble(X, expansion_size=0, batch_size=1000, output_device='cpu') # (num_candidate_points, num_samples)
                    preds = preds.transpose(0, 1)
                    #print("done predictions")

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
                        ei_idx = ei_sortidx[-params.mves_diversity*ack_batch_size:]
                        best_pred = torch.cat([preds[:, ei_idx], preds[:, pdts_idx].unsqueeze(-1)], dim=-1)
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

                    train_Y_mves = (train_Y_mves-Y_mean)/Y_std

                    data = [train_X_mves, train_Y_mves, val_X, val_Y]
                    logging, optim = dbopt.train(
                        params,
                        params.retrain_batch_size, 
                        params.retrain_lr, 
                        params.retrain_num_epochs, 
                        params.hsic_retrain_lambda, 
                        params.num_retrain_latent_samples, 
                        data, 
                        model_mves, 
                        qz_mves, 
                        e_dist)

                    print('logging:', [k[-1] for k in logging])
                    f.write(str([k[-1] for k in logging]) + "\n")
                    logging = [torch.tensor(k) for k in logging]

                    ack_array = np.array(list(ack_all_mves), dtype=np.int32)

                    print('best so far:', labels[ack_array].max())
                    e = reparam.generate_prior_samples(params.ack_num_hsic_samples, e_dist)
                    model_ensemble = reparam.generate_ensemble_from_stochastic_net(model_mves, qz_mves, e)
                    preds = model_ensemble(X, expansion_size=0, batch_size=1000, output_device='cpu') # (num_candidate_points, num_samples)
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
                        'qz_state_dict': qz_mves.state_dict(),
                        'logging': logging,
                        'optim': optim.state_dict(),
                        'ack_idx': torch.from_numpy(ack_array),
                        'ack_labels': torch.from_numpy(labels[ack_array]),
                        'best_hsic': best_hsic,
                        'diversity': len(set(sorted_preds_idx[:, -top_k:].flatten())),
                        'ir_batch_ei': torch.from_numpy(ei),
                        'ir_batch_ei_idx': torch.from_numpy(ei_sortidx),
                        'idx_frac': torch.tensor(idx_frac),
                        'ei_idx': torch.tensor(ei_idx)
                        }, batch_ack_output_file)
                    
            main_f.write(str(ack_batch_size) + "\t" + s + "\n")
