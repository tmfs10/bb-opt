
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

param_names = ["<init_train_epochs>", "<init_lr>", "<retrain_epochs>", "<retrain_lr>", "<ack batch size>", "<ei diversity measure>", "<ucb>", "<unseen reg>", "<gamma>", "<data dir>", "<filename file>", "<output dir>", "<suffix>"]

if len(sys.argv) < len(param_names)+1:
    print("Usage: python", sys.argv[0], " ".join(param_names))
    sys.exit(1)

init_train_epochs, train_lr, retrain_num_epochs, retrain_lr, ack_batch_size, ei_diversity_measure, ucb, unseen_reg, gamma, data_dir, filename_file, output_dir, suffix = sys.argv[1:]

if output_dir[-1] == "/":
    output_dir = output_dir[:-1]
output_dir = output_dir + "_" + suffix

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
    'unseen_reg',
    'gamma',
    
    'num_train_latent_samples',
    'num_latent_vars',
    'train_batch_size', 
    
    'ack_batch_size',
    'num_acks',
    'mves_kernel_fn',
    'input_opt_num_iter',
    'ack_num_model_samples',
    'ei_diversity',
    
    # ensemble
    'num_hidden',
    'num_models',
    'train_l2',
    'retrain_l2',
    
    'retrain_num_epochs',
    'retrain_batch_size',
    'retrain_lr',
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
    unseen_reg=unseen_reg.lower(),
    gamma=float(gamma),
    
    train_batch_size=10,
    num_latent_vars=15,
    num_train_latent_samples=20,
    
    ack_batch_size=int(ack_batch_size),
    num_acks=20,
    mves_kernel_fn='mixrq_kernels',
    input_opt_num_iter=2500,
    ack_num_model_samples=100,
    ei_diversity=ei_diversity_measure.lower(),
    
    num_hidden=100,
    num_models=4,
    train_l2=0.035,
    retrain_l2=0.035,
    
    retrain_num_epochs=int(retrain_num_epochs),
    retrain_batch_size=10,
    retrain_lr=1e-4,
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

    if "pdts" not in params.ei_diversity:
        output_dir2 = output_dir + "_" + str(params.ucb)
    else:
        output_dir2 = output_dir
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

    train_idx, _, _ = utils.train_val_test_split(idx, split=[n_train, 0])
    train_idx2, _, test_idx2 = utils.train_val_test_split(n_train, split=[0.9, 0])

    val_idx = train_idx[test_idx2]
    train_idx = train_idx[train_idx2]

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
            batch_output_dir = main_output_dir + "/" + str(ack_batch_size)
            try:
                os.mkdir(batch_output_dir)
            except OSError as e:
                pass

            train_X_ei = train_X.clone()
            train_Y_ei = train_Y.clone()

            model_ei = copy.deepcopy(model)

            print('doing batch', ack_batch_size)
            skip_idx_ei = set(train_idx)
            ack_all_ei = set()

            if os.path.exists(batch_output_dir + "/" + str(params.num_acks-1) + ".pth"):
                print('already done batch', ack_batch_size)
                continue

            with open(batch_output_dir + "/stats.txt", 'a', buffering=1) as f:
                for ack_iter in range(params.num_acks):
                    batch_ack_output_file = batch_output_dir + "/" + str(ack_iter) + ".pth"
                    if os.path.exists(batch_ack_output_file):
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

                    print('doing ack_iter', ack_iter)
                    model_ensemble = model_ei
                    preds, _ = model_ensemble(X) # (num_candidate_points, num_samples)
                    preds = preds.detach()
                    assert preds.shape[1] == Y.shape[0], "%s[1] == %s[0]" % (str(preds.shape), str(Y.shape))

                    #preds_for_ei = torch.max(preds - train_Y_ei.max(), torch.tensor(0.).to(params.device))
                    #assert preds_for_ei.shape == preds.shape, str(preds_for_ei.shape)

                    idx = list({i for i in range(Y.shape[0])}.difference(skip_idx_ei))
                    #log_prob_list, mse_list = bopt.get_pred_stats(preds, Y.cpu(), train_label_mean, train_label_std, params.output_dist_fn, params.output_dist_std, idx)

                    #print('log_prob_list:', log_prob_list)
                    #print('mse_list:', mse_list)

                    ei = preds.mean(dim=0).view(-1).cpu().numpy()
                    std = preds.std(dim=0).view(-1).cpu().numpy()

                    if "var" in params.ei_diversity:
                        print("var")
                        ei_sortidx = np.argsort(ei/std)
                    elif "ucb" in params.ei_diversity:
                        print("ucb")
                        ei_sortidx = np.argsort(ei + params.ucb*std)
                    else:
                        if "pdts" not in params.ei_diversity:
                            assert False, "Not Implemented: " + params.ei_diversity

                    if "none" in params.ei_diversity:
                        print(filename, "none")
                        ei_idx = []
                        for idx in ei_sortidx[::-1]:
                            if idx not in skip_idx_ei:
                                ei_idx += [idx]
                            if len(ei_idx) >= ack_batch_size:
                                break
                    elif "hsic" in params.ei_diversity:
                        print(filename, "hsic")
                        ei_idx = bopt.ei_diversity_selection_hsic(params, preds, skip_idx_ei, device=params.device)
                    elif "detk" in params.ei_diversity:
                        print(filename, "detk")
                        ei_idx = bopt.ei_diversity_selection_detk(params, preds, skip_idx_ei, device=params.device)
                    elif "pdts" in params.ei_diversity:
                        print(filename, "pdts")
                        ei_idx = set()
                        ei_sortidx = np.argsort(ei)
                        sorted_preds_idx = []
                        for i in range(preds.shape[0]):
                            sorted_preds_idx += [np.argsort(preds[i].numpy())]
                        sorted_preds_idx = np.array(sorted_preds_idx)
                        if "density" in params.ei_diversity:
                            print("pdts_density")
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
                            assert params.ei_diversity == "pdts", params.ei_diversity
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
                        assert False, "Not implemented"
                    assert len(ei_idx) == ack_batch_size, len(ei_idx)

                    best_ei_10 = labels[ei_sortidx[-10:]]
                    f.write('best_ei_10\t' + str(best_ei_10.mean()) + "\t" + str(best_ei_10.max()) + "\t")

                    ack_all_ei.update(ei_idx)
                    skip_idx_ei.update(ei_idx)
                    print("ei_idx:", ei_idx)
                    print('ei_labels', labels[ei_idx])
                    #np.random.shuffle(ei_idx)
                    #print("after ei_idx:", ei_idx)

                    new_idx = list(skip_idx_ei)
                    random.shuffle(new_idx)
                    new_idx = torch.LongTensor(new_idx)

                    train_X_ei = X.new_tensor(X[new_idx])
                    train_Y_ei = Y.new_tensor(Y[new_idx])

                    Y_mean = train_Y_ei.mean()
                    Y_std = train_Y_ei.std()

                    train_label_mean = float(Y_mean.item())
                    train_label_std = float(Y_std.item())

                    train_Y_ei = utils.sigmoid_standardization(train_Y_ei, Y_mean, Y_std, exp=torch.exp)
                    val_Y = utils.sigmoid_standardization(Y[val_idx], Y_mean, Y_std, exp=torch.exp)
                    
                    print("train_X_ei.shape", train_X_ei.shape)
                    
                    expected_num_points = (ack_iter+1)*ack_batch_size
                    assert train_X_ei.shape[0] == int(n_train*0.9) + expected_num_points, str(train_X_ei.shape) + "[0] == " + str(int(n_train*0.9) + expected_num_points)
                    assert train_Y_ei.shape[0] == train_X_ei.shape[0]
                    data = [train_X_ei, train_Y_ei, val_X, val_Y]
                    optim = torch.optim.Adam(list(model_ei.parameters()), lr=params.retrain_lr, weight_decay=params.retrain_l2)
                    logging, optim = dbopt.train_ensemble(
                        params, 
                        params.retrain_batch_size, 
                        params.retrain_num_epochs, 
                        data, 
                        model_ei,
                        optim,
                        unseen_reg=params.unseen_reg,
                        gamma=params.gamma,
                        )

                    print(filename)
                    print('logging:', [k[-1] for k in logging])

                    f.write(str([k[-1] for k in logging]) + "\n")
                    logging = [torch.tensor(k) for k in logging]

                    ack_array = np.array(list(ack_all_ei), dtype=np.int32)

                    print('best so far:', labels[ack_array].max())
                    
                    model_ensemble = model_ei
                    preds, _ = model_ensemble(X) # (num_candidate_points, num_samples)
                    preds = preds.detach()
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

                    torch.save({
                        'model_state_dict': model_ei.state_dict(), 
                        'logging': logging,
                        'optim': optim.state_dict(),
                        'ack_idx': torch.from_numpy(ack_array),
                        'ack_labels': torch.from_numpy(labels[ack_array]),
                        'ir_batch_ei': torch.from_numpy(ei),
                        'ir_batch_ei_idx': torch.from_numpy(ei_sortidx),
                        'idx_frac': torch.tensor(idx_frac),
                        #'test_log_prob': torch.tensor(log_prob_list),
                        #'test_mse': torch.tensor(mse_list),
                        }, batch_ack_output_file)
                    sys.stdout.flush()
                    
            main_f.write(str(ack_batch_size) + "\t" + s + "\n")
