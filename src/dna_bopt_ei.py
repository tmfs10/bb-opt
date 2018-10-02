
import sys
sys.path.append('/cluster/sj1')
sys.path.append('/cluster/sj1/bb_opt/src')

import os
import torch
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

if len(sys.argv) < 13:
    print("Usage: python", sys.argv[0], "<num_diversity> <init_train_epochs> <init_lr> <retrain_epochs> <retrain_lr> <ack batch size> <ei diversity measure> <ucb> <data dir> <filename file> <output dir> <suffix>")
    sys.exit(1)

num_diversity, init_train_epochs, train_lr, retrain_num_epochs, retrain_lr, ack_batch_size, ei_diversity_measure, ucb, data_dir, filename_file, output_dir, suffix = sys.argv[1:]

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
    'ucb',
    
    'num_train_latent_samples',
    'num_latent_vars',
    'train_batch_size', 
    'hsic_train_lambda',
    
    'ack_batch_size',
    'num_acks',
    'mves_kernel_fn',
    'input_opt_num_iter',
    'ack_num_model_samples',
    'hsic_diversity_lambda',
    'mves_compute_batch_size',
    'mves_diversity',
    'ei_diversity',
    
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
    ucb=float(ucb),
    
    train_batch_size=10,
    num_latent_vars=15,
    num_train_latent_samples=20,
    hsic_train_lambda=20.,
    
    ack_batch_size=int(ack_batch_size),
    num_acks=20,
    mves_kernel_fn='mixrq_kernels',
    input_opt_num_iter=2500,
    ack_num_model_samples=100,
    hsic_diversity_lambda=1,
    mves_compute_batch_size=3000,
    mves_diversity=int(num_diversity),
    ei_diversity=ei_diversity_measure.lower(),
    
    retrain_num_epochs=int(retrain_num_epochs),
    retrain_batch_size=10,
    retrain_lr=1e-4,
    hsic_retrain_lambda=20.,
    num_retrain_latent_samples=20,
)

n_train = 20

filenames = [k.strip() for k in open(filename_file).readlines()]

for filename in filenames:
    filedir = data_dir + "/" + filename + "/"
    if not os.path.exists(filedir):
        continue
    print('doing file:', filedir)
    inputs = np.load(filedir+"inputs.npy")
    labels = np.load(filedir+"labels.npy")

    output_dir2 = output_dir + "_" + str(params.ucb)
    main_output_dir = output_dir2 + "/" + filename

    if not os.path.exists(output_dir2):
        os.mkdir(output_dir2)

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

    model, qz, e_dist = dbopt.get_model_nn(params, inputs.shape[1], params.num_latent_vars, params.prior_std)
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

        for ack_batch_size in [2, 5, 10, 20]:
            batch_output_dir = main_output_dir + "/" + str(ack_batch_size)
            try:
                os.mkdir(batch_output_dir)
            except OSError as e:
                pass

            train_X_ei = train_X.clone()
            train_Y_ei = train_Y.clone()

            model_ei = copy.deepcopy(model)
            qz_ei = copy.deepcopy(qz)

            print('doing batch', ack_batch_size)
            skip_idx_ei = set(train_idx)
            ack_all_ei = set()
            e = reparam.generate_prior_samples(params.ack_num_model_samples, e_dist)

            if os.path.exists(batch_output_dir + "/" + str(params.num_acks-1) + ".pth"):
                print('alread done batch', ack_batch_size)
                continue

            with open(batch_output_dir + "/stats.txt", 'a') as f:
                for ack_iter in range(params.num_acks):
                    batch_ack_output_file = batch_output_dir + "/" + str(ack_iter) + ".pth"
                    if os.path.exists(batch_ack_output_file):
                        checkpoint = torch.load(batch_ack_output_file)
                        model_ei.load_state_dict(checkpoint['model_state_dict'])
                        qz_ei.load_state_dict(checkpoint['qz_state_dict'])
                        logging = checkpoint['logging']

                        model_parameters = model_ei.parameters() + qz_ei.parameters()
                        optim = torch.optim.Adam(model_parameters, lr=params.retrain_lr)
                        optim = optim.load_state_dict(checkpoint['optim'])

                        ack_idx = list(checkpoint['ack_idx'].numpy())
                        ack_all_ei.update(ack_idx)
                        skip_idx_ei.update(ack_idx)
                        continue

                    print('doing ack_iter', ack_iter)
                    model_ensemble = reparam.generate_ensemble_from_stochastic_net(model_ei, qz_ei, e)
                    preds = model_ensemble(X, expansion_size=0, batch_size=1000, output_device='cpu') # (num_candidate_points, num_samples)
                    preds = preds.transpose(0, 1)

                    top_k = max(params.mves_diversity, ack_batch_size)

                    ei = preds.mean(dim=0).view(-1).cpu().numpy()
                    std = preds.std(dim=0).view(-1).cpu().numpy()

                    if "var" in params.ei_diversity:
                        ei_sortidx = np.argsort(ei/std)
                    elif "ucb" in params.ei_diversity:
                        ei_sortidx = np.argsort(ei + params.ucb*std)
                    else:
                        ei_sortidx = np.argsort(ei)

                    if "none" in params.ei_diversity:
                        ei_idx = []
                        for idx in ei_sortidx[::-1]:
                            if idx not in skip_idx_ei:
                                ei_idx += [idx]
                            if len(ei_idx) >= top_k:
                                break
                        ei_idx = np.random.choice(ei_idx, ack_batch_size).tolist()
                    elif "hsic" in params.ei_diversity:
                        ei_idx = bopt.ei_diversity_selection_hsic(params, preds, skip_idx_ei, device=params.device)
                    elif "detk" in params.ei_diversity:
                        ei_idx = bopt.ei_diversity_selection_detk(params, preds, skip_idx_ei, device=params.device)
                    else:
                        assert False, "Not implemented"

                    best_ei_10 = labels[ei_sortidx[-10:]]
                    f.write('best_ei_10\t' + str(best_ei_10.mean()) + "\t" + str(best_ei_10.max()))

                    ack_all_ei.update(ei_idx)
                    skip_idx_ei.update(ei_idx)
                    ei_idx = torch.LongTensor(ei_idx).to(params.device)
                    
                    ack_ei = X[ei_idx]
                    ack_ei_vals = (Y[ei_idx]-float(train_label_mean))/float(train_label_std)
                    
                    train_X_ei = torch.cat([train_X_ei, ack_ei], dim=0)
                    train_Y_ei = torch.cat([train_Y_ei, ack_ei_vals], dim=0)
                    data = [train_X_ei, train_Y_ei, val_X, val_Y]
                    logging, optim = dbopt.train(
                        params, 
                        params.retrain_batch_size, 
                        params.retrain_lr,
                        params.retrain_num_epochs, 
                        params.hsic_retrain_lambda, 
                        params.num_retrain_latent_samples, 
                        data, 
                        model_ei, 
                        qz_ei, 
                        e_dist)

                    print([k[-1] for k in logging])

                    print('logging:', [k[-1] for k in logging])
                    f.write(str([k[-1] for k in logging]) + "\n")
                    logging = [torch.tensor(k) for k in logging]

                    ack_array = np.array(list(ack_all_ei), dtype=np.int32)
                    
                    torch.save({
                        'model_state_dict': model_ei.state_dict(), 
                        'qz_state_dict': qz_ei.state_dict(),
                        'logging': logging,
                        'optim': optim.state_dict(),
                        'ack_idx': torch.from_numpy(ack_array),
                        'ack_labels': torch.from_numpy(labels[ack_array]),
                        }, batch_ack_output_file)
                    
                    e = reparam.generate_prior_samples(params.ack_num_model_samples, e_dist)
                    model_ensemble = reparam.generate_ensemble_from_stochastic_net(model_ei, qz_ei, e)
                    preds = model_ensemble(X, expansion_size=0, batch_size=1000, output_device='cpu') # (num_candidate_points, num_samples)
                    preds = preds.transpose(0, 1)
                    ei = preds.mean(dim=0).view(-1).cpu().numpy()
                    ei_sortidx = np.argsort(ei)[-50:]
                    ack = list(ack_all_ei.union(list(ei_sortidx)))
                    best_10 = np.sort(labels[ack])[-10:]
                    best_batch_size = np.sort(labels[ack])[-ack_batch_size:]

                    idx_frac = [len(ack_all_ei.intersection(k))/len(k) for k in top_idx]

                    s = "\t".join([str(k) for k in [
                        best_10.mean(), 
                        best_10.max(), 
                        best_batch_size.mean(), 
                        best_batch_size.max(),
                        str(idx_frac)
                        ]])

                    print(s)
                    f.write(s + "\n")
            main_f.write(str(ack_batch_size) + "\t" + s + "\n")
