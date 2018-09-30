
import sys
sys.path.append('/cluster/sj1')
sys.path.append('/cluster/sj1/bb_opt/src')

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

if len(sys.argv) < 6:
    print("Usage: python", sys.argv[0], "<num_diversity> <init_train_epochs> <init_lr> <retrain_epochs> <ack_batch_size>")
    sys.exit(1)

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
    
    'ack_batch_size',
    'ack_lr',
    'num_queries',
    'batch_opt_lr',
    'batch_opt_num_iter',
    'input_opt_lr',
    'mves_kernel_fn',
    'input_opt_num_iter',
    'ack_num_model_samples',
    'ack_num_pdts_points',
    'hsic_diversity_lambda',
    'mves_compute_batch_size',
    'pdts_diversity',
    
    'retrain_num_epochs',
    'retrain_batch_size',
    'retrain_lr',
    'hsic_retrain_lambda',
    'num_retrain_latent_samples',
    
    'score_fn', # order is logp, qed, sas
    
])
params = Params(
    output_dist_std=1.,
    output_dist_fn=tdist.Normal, 
    exp_noise_samples=2, 
    prior_mean=0.,
    prior_std=3.,
    device='cuda',
    num_epochs=int(sys.argv[2]),
    train_lr=float(sys.argv[3]),
    
    train_batch_size=10,
    num_latent_vars=15,
    num_train_latent_samples=20,
    hsic_train_lambda=20.,
    
    ack_batch_size=int(sys.argv[5]),
    ack_lr=1e-3,
    num_queries=1,
    batch_opt_lr=1e-3,
    batch_opt_num_iter=10000,
    input_opt_lr=1e-3,
    mves_kernel_fn='mixrq_kernels',
    input_opt_num_iter=2500,
    ack_num_model_samples=100,
    ack_num_pdts_points=40,
    hsic_diversity_lambda=1,
    mves_compute_batch_size=3000,
    pdts_diversity=int(sys.argv[1]),
    
    retrain_num_epochs=int(sys.argv[4]),
    retrain_batch_size=10,
    retrain_lr=1e-4,
    hsic_retrain_lambda=20.,
    num_retrain_latent_samples=20,
    
    score_fn=lambda x : 5*x[1]-x[2],
)

n_train = 20

project = "dna_binding"
dataset = "crx_ref_r1"

root = "/cluster/sj1/bb_opt/"
data_dir = root+"data/"+project+"/"+dataset+"/"
inputs = np.load(data_dir+"inputs.npy")
labels = np.load(data_dir+"labels.npy")


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
top_point_idx = labels_sort_idx[-1]
top_one_percent_idx = labels_sort_idx[-int(labels.shape[0]*0.01):]
top_one_percent_sum = labels[top_one_percent_idx].sum()

train_label_mean = train_labels.mean()
train_label_std = train_labels.std()

train_labels = (train_labels - train_label_mean) / train_label_std
val_labels = (val_labels - train_label_mean) / train_label_std

train_X_pdts = torch.FloatTensor(train_inputs).to(device)
train_Y_pdts = torch.FloatTensor(train_labels).to(device)
val_X = torch.FloatTensor(val_inputs).to(device)
val_Y = torch.FloatTensor(val_labels).to(device)

model, qz, e_dist = dbopt.get_model_nn(params, inputs.shape[1], params.num_latent_vars, params.prior_std)
data = [train_X_pdts, train_Y_pdts, val_X, val_Y]
logging = dbopt.train(params, params.train_batch_size, params.train_lr, params.num_epochs, params.hsic_train_lambda, params.num_train_latent_samples, data, model, qz, e_dist)

print([k[-1] for k in logging])

e = reparam.generate_prior_samples(params.ack_num_model_samples, e_dist)

X = torch.tensor(inputs, device=device)
Y = torch.tensor(labels, device=device)

model_pdts = model
qz_pdts = qz

skip_idx_pdts = set(train_idx)
ack_all_pdts = set()
for ack_iter in range(10):
    model_ensemble = reparam.generate_ensemble_from_stochastic_net(model_pdts, qz_pdts, e)
    preds = model_ensemble(X, expansion_size=0, batch_size=1000, output_device='cpu') # (num_candidate_points, num_samples)
    preds = preds.transpose(0, 1)
    print("done predictions")

    preds[:, list(skip_idx_pdts)] = preds.min()
    
    top_k = max(params.pdts_diversity, params.ack_batch_size)
    sorted_preds_idx = []
    for i in range(preds.shape[0]):
        sorted_preds_idx += [np.argsort(preds[i].numpy())]
    sorted_preds_idx = np.array(sorted_preds_idx)

    pdts_idx = set()
    counts = np.zeros(sorted_preds_idx.shape[1])
    for rank in range(sorted_preds_idx.shape[1]-1, 0, -1):
        counts[:] = 0
        for idx in sorted_preds_idx[:, rank]:
            counts[idx] += 1
        counts_idx = counts.argsort()[::-1]
        j = 0
        while len(pdts_idx) < top_k and j < counts_idx.shape[0] and counts[counts_idx[j]] > 0:
            pdts_idx.update({counts_idx[j]})
            j += 1
        if len(pdts_idx) >= top_k:
            break
    assert len(pdts_idx) == top_k
    pdts_idx = np.random.choice(list(pdts_idx), params.ack_batch_size)

    ack_all_pdts.update(pdts_idx)
    skip_idx_pdts.update(pdts_idx)
    
    ack_pdts = X[pdts_idx]
    ack_pdts_vals = (Y[pdts_idx]-float(train_label_mean))/float(train_label_std)
    
    train_X_pdts = torch.cat([train_X_pdts, ack_pdts], dim=0)
    train_Y_pdts = torch.cat([train_Y_pdts, ack_pdts_vals], dim=0)
    data = [train_X_pdts, train_Y_pdts, val_X, val_Y]
    logging = dbopt.train(
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

    print([k[-1] for k in logging])
    
    e = reparam.generate_prior_samples(params.ack_num_model_samples, e_dist)
    model_ensemble = reparam.generate_ensemble_from_stochastic_net(model_pdts, qz_pdts, e)
    preds = model_ensemble(X, expansion_size=0, batch_size=1000, output_device='cpu') # (num_candidate_points, num_samples)
    preds = preds.transpose(0, 1)
    ei = preds.mean(dim=0).view(-1).cpu().numpy()
    ei_sortidx = np.argsort(ei)[-50:]
    ack = list(ack_all_pdts.union(list(ei_sortidx)))
    best_10 = np.sort(labels[ack])[-10:]
    print(len(ack), best_10.mean(), best_10.max())