
import sys
import os
from gpu_utils.utils import gpu_init, nvidia_smi
import parsing
import argparse

from collections import namedtuple

def convert_to_namedtuple(d):
    return namedtuple('GenericDict', d.keys())(**d)

sys.path.append('/cluster/sj1')
sys.path.append('/cluster/sj1/bb_opt/src')

parser = argparse.ArgumentParser()
parsing.add_inference_from_saved_model_args(parser)

params = parser.parse_args()

if params.device != "cpu":
    if params.gpu == -1:
        gpu_id = gpu_init(best_gpu_metric="mem")
    else:
        gpu_id = gpu_init(gpu_id=params.gpu)

import os
import torch
import pprint
import random
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.distributions as tdist
import numpy as np
from scipy.stats import kendalltau, pearsonr
import train
import chemvae_bopt as cbopt
import bayesian_opt as bopt
import active_learning as al
import pandas as pd
import copy
import non_matplotlib_utils as utils
import datetime
import ops
from deep_ensemble_sid import NNEnsemble, ResnetEnsemble, ResnetEnsemble2

torch.backends.cudnn.deterministic = True

if params.device != "cpu":
    print("Running on GPU " + str(gpu_id))
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
params.device = device

# main output dir
task_name = []
if params.project == 'dna_binding':
    task_name = [k.strip() for k in open(params.filename_file).readlines()][:params.num_test_tfs]
    sample_uniform_fn = train.dna_sample_uniform
elif params.project in ['imdb', 'wiki']:
    task_name += [params.project]
    sample_uniform_fn = train.image_sample_uniform
elif params.project == 'chemvae':
    task_name += [params.project]
    sample_uniform_fn = None
elif params.project == 'test_fn':
    task_name += [params.test_fn]
    sample_uniform_fn = None

orig_params = params
for task_iter in range(len(task_name)):
    for seed in range(1, orig_params.num_seeds+1):
        print('DOING SEED', seed)
        init_model_path = orig_params.init_model_path_prefix + "_" + str(seed) + "/" + task_name[task_iter] + "/init_model.pth"
        if not os.path.exists(init_model_path):
            print('CAN\'T FIND SEED', seed, '. SKIPPING...')
            continue
        checkpoint = torch.load(init_model_path)
        save_file_path_prefix = orig_params.save_file_path_prefix
        params = checkpoint['global_params']
        params = convert_to_namedtuple(params)

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

        ood_inputs = None
        ood_labels = None
        if params.project == 'dna_binding':
            inputs = np.load(params.data_dir + "/" + task_name[task_iter] + "/inputs.npy").astype(np.float32)
            labels = np.load(params.data_dir + "/" + task_name[task_iter] + "/labels.npy").astype(np.float32)
            if params.take_log:
                labels = np.log(labels)

            X = torch.tensor(inputs, device=device)
            Y = torch.tensor(labels, device=device)
        elif params.project == 'chemvae':
            inputs = np.load(params.data_dir + "/chemvae_inputs.npy").astype(np.float32)
            labels = np.load(params.data_dir + "/chemvae_labels.npy").astype(np.float32)
            if params.take_log:
                labels = np.log(labels)

            X = torch.tensor(inputs, device=device)
            Y = torch.tensor(labels, device=device)
        elif params.project in ['imdb', 'wiki']:
            inputs, labels, gender = utils.load_data_wiki_sid(
                    params.data_dir,
                    params.project,
                    )
            inputs = inputs.astype(np.float32)/255.
            male_gender = (gender == 1)
            female_gender = (gender == 0)
            
            ood_inputs = inputs[female_gender].astype(np.float32)
            ood_labels = labels[female_gender].astype(np.float32)

            #if params.num_ood_to_eval > 0:
            #    rand_idx = np.random.choice(ood_inputs.shape[0], size=(params.num_ood_to_eval,), replace=False).tolist()
            #    ood_inputs = ood_inputs[rand_idx]
            #    ood_labels = ood_labels[rand_idx]

            inputs = inputs[male_gender].astype(np.float32)
            labels = labels[male_gender].astype(np.float32)

            #temp_idx = np.random.choice(inputs.shape[0], size=(2000,), replace=False).tolist()
            #inputs = inputs[temp_idx]
            #labels = labels[temp_idx]

            X = torch.tensor(inputs, device=device)
            Y = torch.tensor(labels, device=device)

            ood_X = torch.tensor(ood_inputs, device=device)
            ood_Y = torch.tensor(ood_labels, device=device)
        elif params.project == "test_fn":
            params.predict_mi = False
            params.infomax_weight = 0
            params.num_test_points = 0
            params.ack_change_stat_logging = False
            params.empirical_ack_change_stat_logging = False
            import test_fns
            bb_fn = getattr(test_fns, params.test_fn)()
            inputs, labels = bb_fn.sample(2*params.init_train_examples)
            inputs = inputs.cpu().numpy()
            labels = labels.cpu().numpy()

            X = torch.tensor(inputs, device=device)
            Y = torch.tensor(labels, device=device)
        else:
            assert False, params.project + " is an unknown project"

        indices = np.arange(labels.shape[0])
        labels_sort_idx = labels.argsort()
        idx_to_rel_opt_value = {labels_sort_idx[i] : float(i+1)/len(labels_sort_idx) for i in range(len(labels_sort_idx))}
        if params.exclude_top > 0.001:
            sort_idx = labels_sort_idx[:-int(labels.shape[0]*params.exclude_top)]
        else:
            sort_idx = labels_sort_idx
        indices = indices[sort_idx]

        # reading data
        train_idx, val_idx, test_idx, data_split_rng = utils.train_val_test_split(
                indices, 
                split=[params.init_train_examples, params.held_out_val],
                rng=data_split_rng,
                )

        train_inputs = inputs[train_idx]
        train_labels = labels[train_idx]

        val_inputs = inputs[val_idx]
        val_labels = labels[val_idx]

        test_idx = test_idx[:params.num_test_points]
        test_inputs = inputs[test_idx]
        test_labels = inputs[test_idx]

        top_frac_idx = [set(labels_sort_idx[-int(labels.shape[0]*per):].tolist()) for per in [0.01, 0.05, 0.1, 0.2]]

        print('label stats:', labels.mean(), labels.max(), labels.std())
        print('num_test_idx:', len(test_idx))

        train_X_init = torch.FloatTensor(train_inputs).to(device)
        train_Y_cur = torch.FloatTensor(train_labels).to(device)

        val_X = None
        val_Y = None
        if params.held_out_val > 0:
            val_X = torch.FloatTensor(val_inputs).to(device)
            val_Y = torch.FloatTensor(val_labels).to(device)

        cur_rng = ops.get_rng_state()
        ops.set_rng_state(ensemble_init_rng)

        if params.project == "dna_binding":
            init_model = train.get_model_nn_ensemble(
                    inputs.shape[1], 
                    params.num_models, 
                    params.num_hidden, 
                    sigmoid_coeff=params.sigmoid_coeff, 
                    device=params.device,
                    separate_mean_var=params.separate_mean_var,
                    mu_prior=params.bayesian_theta_prior_mu if params.bayesian_ensemble else None,
                    std_prior=params.bayesian_theta_prior_std if params.bayesian_ensemble else None,
                    )
        elif params.project == "chemvae":
            model_kwargs = {
                    "params": params, 
                    "num_inputs": inputs.shape[1],
                   }
            init_model = NNEnsemble(
                    params.num_models,
                    cbopt.PropertyPredictor,
                    model_kwargs,
                    device=params.device,
                    )
            init_model = init_model.to(params.device)
        elif params.project == "test_fn":
            init_model = train.get_model_nn_ensemble(
                    inputs.shape[1], 
                    params.num_models, 
                    params.num_hidden, 
                    sigmoid_coeff=params.sigmoid_coeff, 
                    device=params.device,
                    separate_mean_var=params.separate_mean_var,
                    mu_prior=params.bayesian_theta_prior_mu if params.bayesian_ensemble else None,
                    std_prior=params.bayesian_theta_prior_std if params.bayesian_ensemble else None,
                    )
        elif params.project in ["wiki", "imdb"]:
            init_model = ResnetEnsemble2(
                    params,
                    params.num_models, 
                    inputs.shape[1], 
                    depth=params.resnet_depth,
                    widen_factor=params.resnet_width_factor,
                    n_hidden=params.num_hidden,
                    dropout_factor=params.resnet_dropout,
                    device=params.device,
                    mu_prior=params.bayesian_theta_prior_mu if params.bayesian_ensemble else None,
                    std_prior=params.bayesian_theta_prior_std if params.bayesian_ensemble else None,
                    ).to(params.device)
        else:
            assert False, params.project + " project not implemented"

        ensemble_init_rng = ops.get_rng_state()
        ops.set_rng_state(cur_rng)

        skip_idx = set(train_idx.tolist() + test_idx.tolist() + val_idx.tolist())
        unseen_idx = set(range(Y.shape[0])).difference(skip_idx)

        init_model.load_state_dict(checkpoint['model_state_dict'])

        with torch.no_grad():
            init_model.eval()
            pre_ack_pred_means, pre_ack_pred_vars = train.ensemble_forward(
                    init_model, 
                    X, 
                    params.ensemble_forward_batch_size,
                    progress_bar=params.progress_bar,
                    ) # (num_candidate_points, num_samples)

            test_pred_means = pre_ack_pred_means[:, test_idx].cpu().numpy()
            test_pred_vars = pre_ack_pred_vars[:, test_idx].cpu().numpy()

            Y = utils.sigmoid_standardization(
                    Y,
                    train_Y_cur.mean(),
                    train_Y_cur.std(),
                    exp=torch.exp)
            test_Y = Y[test_idx].cpu().numpy()

            np.save(save_file_path_prefix + "_test_pred_means_" + str(seed) + ".npy", test_pred_means)
            np.save(save_file_path_prefix + "_test_pred_vars_" + str(seed) + ".npy", test_pred_vars)
            np.save(save_file_path_prefix + "_test_Y_" + str(seed) + ".npy", test_Y)

        if ood_inputs is not None:
            assert ood_labels is not None
            assert ood_inputs.shape[0] == ood_labels.shape[0]

            with torch.no_grad():
                ood_pred_means, ood_pred_vars = train.ensemble_forward(
                        init_model, 
                        ood_X, 
                        params.ensemble_forward_batch_size,
                        progress_bar=params.progress_bar,
                        ) # (num_candidate_points, num_samples)

                ood_pred_means = ood_pred_means.transpose(0, 1).cpu().numpy()
                ood_pred_vars = ood_pred_vars.transpose(0, 1).cpu().numpy()
                ood_Y = utils.sigmoid_standardization(
                        ood_Y,
                        train_Y_cur.mean(),
                        train_Y_cur.std(),
                        exp=torch.exp)
                ood_Y = ood_Y.cpu().numpy()

                np.save(save_file_path_prefix + "_ood_pred_means_" + str(seed) + ".npy", ood_pred_means)
                np.save(save_file_path_prefix + "_ood_pred_vars_" + str(seed) + ".npy", ood_pred_vars)
                np.save(save_file_path_prefix + "_ood_Y_" + str(seed) + ".npy", ood_Y)
