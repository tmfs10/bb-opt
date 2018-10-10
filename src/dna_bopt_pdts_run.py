
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

if len(sys.argv) < 11:
    print("Usage: python", sys.argv[0], "<num_diversity> <init_train_epochs> <init_lr> <retrain_epochs> <retrain_lr> <ack_batch_size> <data dir> <filename file> <output dir> <suffix>")
    sys.exit(1)

num_diversity, init_train_epochs, train_lr, retrain_num_epochs, retrain_lr, ack_batch_size, pdts_density, data_dir, filename_file, output_dir, suffix = sys.argv[1:]

if output_dir[-1] == "/":
    output_dir = output_dir[:-1]
output_dir = output_dir + "_pdts_dna_" + suffix

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
    
    'num_train_latent_samples',
    'num_latent_vars',
    'train_batch_size', 
    'hsic_train_lambda',
    'pdts_density',
    
    'ack_batch_size',
    'num_acks',
    'pdts_kernel_fn',
    'ack_num_model_samples',
    'ack_num_pdts_points',
    'hsic_diversity_lambda',
    'pdts_compute_batch_size',
    'pdts_diversity',
    
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
    pdts_density=int(pdts_density) == 1,
    
    train_batch_size=10,
    num_latent_vars=15,
    num_train_latent_samples=20,
    hsic_train_lambda=20.,
    
    ack_batch_size=int(ack_batch_size),
    num_acks=20,
    pdts_kernel_fn='mixrq_kernels',
    ack_num_model_samples=100,
    ack_num_pdts_points=40,
    hsic_diversity_lambda=1,
    pdts_compute_batch_size=4000,
    pdts_diversity=int(num_diversity),
    
    retrain_num_epochs=int(retrain_num_epochs),
    retrain_batch_size=10,
    retrain_lr=float(retrain_lr),
    hsic_retrain_lambda=20.,
    num_retrain_latent_samples=20,
)

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

    model, qz, e_dist = dbopt.get_model_nn(params.prior_mean, params.prior_std, inputs.shape[1], params.num_latent_vars, params.prior_std)
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

            train_X_pdts = train_X.clone()
            train_Y_pdts = train_Y.clone()

            model_pdts = copy.deepcopy(model)
            qz_pdts = copy.deepcopy(qz)

            skip_idx_pdts = set(train_idx)
            ack_all_pdts = set()
            e = reparam.generate_prior_samples(params.ack_num_model_samples, e_dist)

            if os.path.exists(batch_output_dir + "/" + str(params.num_acks-1) + ".pth"):
                print('already done batch', ack_batch_size)
                continue

            with open(batch_output_dir + "/stats.txt", 'a') as f:
                for ack_iter in range(params.num_acks):
                    batch_ack_output_file = batch_output_dir + "/" + str(ack_iter) + ".pth"
                    if os.path.exists(batch_ack_output_file):
                        checkpoint = torch.load(batch_ack_output_file)
                        model_pdts.load_state_dict(checkpoint['model_state_dict'])
                        qz_pdts.load_state_dict(checkpoint['qz_state_dict'])
                        logging = checkpoint['logging']

                        model_parameters = list(model_pdts.parameters()) + list(qz_pdts.parameters())
                        optim = torch.optim.Adam(model_parameters, lr=params.retrain_lr)
                        optim = optim.load_state_dict(checkpoint['optim'])

                        ack_idx = list(checkpoint['ack_idx'].numpy())
                        ack_all_pdts.update(ack_idx)
                        skip_idx_pdts.update(ack_idx)
                        continue

                    print('doing ack_iter', ack_iter, 'for pdts with suffix', suffix)
                    model_ensemble = reparam.generate_ensemble_from_stochastic_net(model_pdts, qz_pdts, e)
                    preds = model_ensemble(X, expansion_size=0, batch_size=1000, output_device='cpu') # (num_candidate_points, num_samples)
                    preds = preds.transpose(0, 1)
                    preds = preds[:ack_batch_size, :]

                    preds[:, list(skip_idx_pdts)] = preds.min()
                    sorted_preds, sorted_preds_idx = torch.sort(preds, dim=1, descending=True)
                    sorted_preds_idx = sorted_preds_idx.cpu().numpy()

                    top_k = params.pdts_diversity
                    
                    pdts_idx = set()
                    if params.pdts_density:
                        counts = np.zeros(sorted_preds_idx.shape[1])
                        for rank in range(sorted_preds_idx.shape[1]):
                            counts[:] = 0
                            for idx in sorted_preds_idx[:, rank]:
                                counts[idx] += 1
                            counts_idx = counts.argsort()[::-1]
                            j = 0
                            while len(pdts_idx) < ack_batch_size and j < counts_idx.shape[0] and counts[counts_idx[j]] > 0:
                                pdts_idx.update({counts_idx[j]})
                                j += 1
                            if len(pdts_idx) >= ack_batch_size:
                                break
                    else:
                        for i_model in range(sorted_preds_idx.shape[0]):
                            for idx in sorted_preds_idx[i_model]:
                                if idx not in pdts_idx:
                                    pdts_idx.update({idx})
                                    break
                            if len(pdts_idx) >= ack_batch_size:
                                break
                    pdts_idx = list(pdts_idx)
                    assert len(pdts_idx) == ack_batch_size

                    ack_all_pdts.update(pdts_idx)
                    skip_idx_pdts.update(pdts_idx)

                    new_idx = list(skip_idx_pdts)
                    random.shuffle(new_idx)
                    new_idx = torch.LongTensor(new_idx)
                    
                    ack_pdts = X[pdts_idx]
                    ack_pdts_vals = (Y[pdts_idx]-float(train_label_mean))/float(train_label_std)

                    train_X_pdts = X.new_tensor(X[new_idx])
                    train_Y_pdts = Y.new_tensor(Y[new_idx])

                    Y_mean = train_Y_pdts.mean()
                    Y_std = train_Y_pdts.std()

                    train_Y_pdts = (train_Y_pdts-Y_mean)/Y_std

                    data = [train_X_pdts, train_Y_pdts, val_X, val_Y]
                    logging, optim = dbopt.train(
                        params, 
                        params.retrain_batch_size, 
                        params.retrain_lr, 
                        params.retrain_num_epochs, 
                        params.hsic_retrain_lambda, 
                        params.num_retrain_latent_samples, 
                        data, 
                        model_pdts, 
                        qz_pdts, 
                        e_dist)

                    print('logging:', [k[-1] for k in logging])
                    f.write(str([k[-1] for k in logging]) + "\n")
                    logging = [torch.tensor(k) for k in logging]

                    ack_array = np.array(list(ack_all_pdts), dtype=np.int32)

                    torch.save({
                        'model_state_dict': model_pdts.state_dict(), 
                        'qz_state_dict': qz_pdts.state_dict(),
                        'logging': logging,
                        'optim': optim.state_dict(),
                        'ack_idx': torch.from_numpy(ack_array),
                        'ack_labels': torch.from_numpy(labels[ack_array]),
                        'diversity': len(set(sorted_preds_idx[:, -top_k:].flatten())),
                        }, batch_ack_output_file)
                    
                    print('best so far:', labels[ack_array].max())
                    e = reparam.generate_prior_samples(100, e_dist)
                    model_ensemble = reparam.generate_ensemble_from_stochastic_net(model_pdts, qz_pdts, e)
                    preds = model_ensemble(X, expansion_size=0, batch_size=1000, output_device='cpu') # (num_candidate_points, num_samples)
                    preds = preds.transpose(0, 1)
                    ei = preds.mean(dim=0).view(-1).cpu().numpy()
                    ei_sortidx = np.argsort(ei)[-50:]
                    ack = list(ack_all_pdts.union(list(ei_sortidx)))
                    best_10 = np.sort(labels[ack])[-10:]
                    best_batch_size = np.sort(labels[ack])[-ack_batch_size:]

                    idx_frac = [len(ack_all_pdts.intersection(k))/len(k) for k in top_idx]

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
