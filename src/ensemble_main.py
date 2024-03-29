
import sys
from gpu_utils.utils import gpu_init, nvidia_smi
import parsing
import argparse

parser = argparse.ArgumentParser()
parsing.add_parse_args(parser)
parsing.add_parse_args_wrongness(parser)

params = parsing.parse_args(parser)

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

do_model_hparam_search = len(params.gammas) > 1

if params.device != "cpu":
    print("Running on GPU " + str(gpu_id))
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
params.device = device

# set data split seed to model seed if data split seed is -1
if params.model_init_seed == -1:
    params.model_init_seed = params.seed
if params.data_split_seed == -1:
    params.data_split_seed = params.model_init_seed
if params.re_train_num_epochs == -1:
    params.re_train_num_epochs = params.init_train_num_epochs
if params.re_train_lr < 0:
    params.re_train_lr = params.init_train_lr
if params.re_train_batch_size == -1:
    params.re_train_batch_size = params.init_train_batch_size
if params.re_train_l2 < 0:
    params.re_train_l2 = params.init_train_l2

print('PARAMS:')
for k, v in vars(params).items():
    print(k, v)

np.random.seed(params.seed)
torch.manual_seed(params.seed)
random.seed(params.seed)

# main output dir
random_label = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
if params.output_dir[-1] == "/":
    params.output_dir = params.output_dir[:-1]
params.output_dir = params.output_dir + "_" + params.ack_fun + "_" + params.suffix

if not os.path.exists(params.output_dir):
    os.mkdir(params.output_dir)

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

if params.stdout_file != "stdout":
    orig_stdout = sys.stdout
    sys.stdout = open(params.output_dir + '/stdout.txt', 'w')

print('output_dir:', params.output_dir)

for task_iter in range(len(task_name)):

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
        
        #ood_inputs = inputs[female_gender].astype(np.float32)
        #ood_labels = labels[female_gender].astype(np.float32)

        #if params.num_ood_to_eval > 0:
        #    rand_idx = np.random.choice(ood_inputs.shape[0], size=(params.num_ood_to_eval,), replace=False).tolist()
        #    ood_inputs = ood_inputs[rand_idx]
        #    ood_labels = ood_labels[rand_idx]

        inputs = inputs[male_gender].astype(np.float32)
        labels = labels[male_gender].astype(np.float32)

        temp_sortidx = np.argsort(labels)
        ind_idx = temp_sortidx[:-int(labels.shape[0]*0.1)]
        ood_idx = temp_sortidx[-int(labels.shape[0]*0.1):]
        ood_inputs = inputs[ood_idx]
        ood_labels = labels[ood_idx]
        inputs = inputs[ind_idx]
        labels = labels[ind_idx]

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
        inputs, labels = bb_fn.sample(params.test_fn_dataset_size)
        inputs = inputs.cpu().numpy()
        labels = labels.cpu().numpy()

        X = torch.tensor(inputs, device=device)
        Y = torch.tensor(labels, device=device)
    else:
        assert False, params.project + " is an unknown project"

    sorted_labels_idx = np.argsort(labels)[::-1]
    sorted_labels_idx_to_rank = {sorted_labels_idx[i] : i for i in range(len(sorted_labels_idx))}

    # file output dir
    file_output_dir = params.output_dir + "/" + task_name[task_iter]
    try:
        os.mkdir(file_output_dir)
    except OSError as e:
        pass

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

    print('label stats:', labels.mean(), labels.min(), labels.max(), labels.std())
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
    untrained_model = copy.deepcopy(init_model)

    ensemble_init_rng = ops.get_rng_state()
    ops.set_rng_state(cur_rng)

    predict_info_models = None
    if params.predict_mi:
        predict_kernel_dim = 10
        predict_info_models = train.PredictInfoModels(
                inputs.shape[1],
                params.ack_emb_kernel_dim,
                params.num_models,
                predict_kernel_dim,
                params.predict_stddev,
                params.predict_mmd,
                params.predict_nll,
                ).to(params.device)

    skip_idx = set(train_idx.tolist() + test_idx.tolist() + val_idx.tolist())
    unseen_idx = set(range(Y.shape[0])).difference(skip_idx)

    init_model_path = file_output_dir + "/init_model.pth"
    loaded = False

    if os.path.isfile(init_model_path) and not params.clean:
        loaded = True
        checkpoint = torch.load(init_model_path)
        init_model.load_state_dict(checkpoint['model_state_dict'])
        logging = checkpoint["logging"]
        zero_gamma_model = None
        best_gamma = -1
        if "train_idx" in checkpoint:
            train_idx = checkpoint["train_idx"].numpy()
    else:
        if params.project in ['imdb', 'wiki']:
            if params.sampling_space == "ood":
                def sample_uniform_fn(batch_size, sampling_info=None):
                    ood_sampling_rng = sampling_info['ood_sampling_rng']
                    cur_rng = ops.get_rng_state()
                    ops.set_rng_state(ood_sampling_rng)
                    idx = [i for i in range(ood_X.shape[0])]
                    np.random.shuffle(idx)
                    sampling_info['ood_sampling_rng'] = ops.get_rng_state()
                    ops.set_rng_state(cur_rng)
                    out = ood_X[idx[:batch_size]]
                    return out
            elif params.sampling_space == "unseen_ind":
                def sample_uniform_fn(batch_size, sampling_info=None):
                    ood_sampling_rng = sampling_info['ood_sampling_rng']
                    cur_rng = ops.get_rng_state()
                    ops.set_rng_state(ood_sampling_rng)
                    idx = list(set(range(X.shape[0])).difference(set(train_idx.tolist() + test_idx.tolist())))
                    np.random.shuffle(idx)
                    sampling_info['ood_sampling_rng'] = ops.get_rng_state()
                    ops.set_rng_state(cur_rng)
                    out = X[idx[:batch_size]]
                    return out
            elif params.sampling_space == "input" and params.sampling_dist == "uniform":
                def sample_uniform_fn(batch_size, sampling_info=None):
                    out, rng = train.image_sample_uniform(batch_size, sampling_info["ood_sampling_rng"])
                    sampling_info["ood_sampling_rng"] = rng
                    return out
            elif params.sampling_space == "input" and params.sampling_dist == "boundary":
                def sample_uniform_fn(batch_size, sampling_info=None):
                    perturb, rng = train.image_sample_gaussian(batch_size, sampling_info["ood_sampling_rng"], scale=params.boundary_stddev)
                    sampling_info["ood_sampling_rng"] = rng
                    with torch.no_grad():
                        images = sampling_info["bX"]
                        assert images.shape[0] == batch_size
                        images = images + perturb
                        images = torch.clamp(images, images.min(), images.max())
                    return images
            elif params.sampling_space == "fc" and params.sampling_dist == "uniform":
                def sample_uniform_fn(batch_size, sampling_info=None):
                    model = sampling_info['model']
                    fc_input_size = model.fc_input_size()
                    dist = tdist.uniform.Uniform(torch.tensor(0.0), torch.tensor(1.0))
                    out = dist.sample(sample_shape=(torch.Size([batch_size, fc_input_size])))
                    return out.to(params.device)
            elif params.sampling_space == "input" and params.sampling_dist == "uniform_bb": # bb = bounding box
                def sample_uniform_fn(batch_size, sampling_info):
                    max_px = sampling_info['max_px']
                    min_px = sampling_info['min_px']
                    u = tdist.Uniform(min_px, max_px)
                    return u.sample(sample_shape=torch.Size([batch_size])).view([batch_size]+list(X.shape[1:])).to(params.device)
            elif params.sampling_space == "fc" and params.sampling_dist == "pom": # pom = product of marginals
                def sample_uniform_fn(batch_size, sampling_info):
                    hist = sampling_info['hist']
                    out = []
                    for i_model in range(len(hist)):
                        out += [ [] ]
                        for i_point in range(len(hist[i_model])):
                            n = len(hist[i_model][i_point][0])
                            idx = np.random.choice(n, size=batch_size, p=hist[i_model][i_point][0])
                            low = hist[i_model][i_point][1][idx]
                            high = hist[i_model][i_point][1][idx+1]
                            out[-1] += [torch.from_numpy(np.random.uniform(low=low, high=high).astype(np.float32)).to(params.device)]
                        out[-1] = torch.stack(out[-1], dim=0).transpose(0, 1)
                    return torch.stack(out, dim=0)
            elif params.sampling_space == "fc" and params.sampling_dist == "uniform_bb": # pom = product of marginals
                def sample_uniform_fn(batch_size, sampling_info):
                    max_val = sampling_info['max_val']
                    min_val = sampling_info['min_val']
                    out = []
                    ood_sampling_rng = sampling_info['ood_sampling_rng']
                    cur_rng = ops.get_rng_state()
                    ops.set_rng_state(ood_sampling_rng)
                    for i in range(len(max_val)):
                        u = tdist.uniform.Uniform(min_val[i], max_val[i])
                        out += [u.sample(sample_shape=torch.Size([batch_size]))]
                    sampling_info['ood_sampling_rng'] = ops.get_rng_state()
                    ops.set_rng_state(cur_rng)
                    return torch.stack(out, dim=0)
            elif params.sampling_space == "indist" and params.sampling_dist == "uniform":
                def sample_uniform_fn(batch_size, sampling_info):
                    train_x = sampling_info["train_x"]
                    ood_sampling_rng = sampling_info["ood_sampling_rng"]
                    cur_rng = ops.get_rng_state()
                    ops.set_rng_state(ood_sampling_rng)
                    idx = [i for i in range(train_x.shape[0])]
                    np.random.shuffle(idx)
                    sampling_info['ood_sampling_rng'] = ops.get_rng_state()
                    ops.set_rng_state(cur_rng)
                    out = train_x[idx[:batch_size]]
                    return out
            else:
                assert False, params.sampling_dist + " not implemented"

        zero_gamma_model = None
        if params.project in ['imdb', 'wiki']:
            logging, best_gamma, data_split_rng, zero_logging, zero_gamma_model, init_model = train.hyper_param_train(
                params,
                init_model,
                [train_X_init, train_Y_cur, val_X, val_Y, X, Y],
                "init",
                params.gammas,
                params.unseen_reg,
                data_split_rng=data_split_rng,
                predict_info_models=predict_info_models if params.predict_mi else None,
                sample_uniform_fn=sample_uniform_fn,
                report_zero_gamma=params.report_zero_gamma,
                normalize_fn=utils.sigmoid_standardization if params.sigmoid_coeff > 0 else utils.normal_standardization,
                num_epoch_iters=None if params.fixed_num_epoch_iters == 0 else params.fixed_num_epoch_iters,
                unseen_idx=unseen_idx,
                mod_adversarial_test=params.mod_adversarial_test,
                hyper_params=[['adv_epsilon', 'langevin_lr', 'langevin_num_iter', 'langevin_beta'], [params.adv_epsilon, params.langevin_lr, params.langevin_num_iter, params.langevin_beta]],
                )
        else:
            logging, best_gamma, data_split_rng, zero_logging, zero_gamma_model, init_model = train.hyper_param_train(
                params,
                init_model,
                [train_X_init, train_Y_cur, val_X, val_Y, X, Y],
                "init",
                params.gammas,
                params.unseen_reg,
                data_split_rng=data_split_rng,
                predict_info_models=predict_info_models if params.predict_mi else None,
                sample_uniform_fn=sample_uniform_fn,
                normalize_fn=utils.sigmoid_standardization if params.sigmoid_coeff > 0 else utils.normal_standardization,
                num_epoch_iters=None if params.fixed_num_epoch_iters == 0 else params.fixed_num_epoch_iters,
                unseen_idx=unseen_idx,
                mod_adversarial_test=params.mod_adversarial_test,
                hyper_params=[['adv_epsilon', 'langevin_lr', 'langevin_num_iter', 'langevin_beta'], [params.adv_epsilon, params.langevin_lr, params.langevin_num_iter, params.langevin_beta]],
                )
        torch.cuda.empty_cache()
        #print('point 6:', nvidia_smi())

        if logging is not None and logging[1] is not None and 'val' in logging[1] and 'rmse' in logging[1]['val']:
            cur_rmse = logging[1]['val']['rmse']
        else:
            cur_rmse = None

    with torch.no_grad():
        init_model.eval()
        pre_ack_pred_means, pre_ack_pred_vars = train.ensemble_forward(
                init_model, 
                X, 
                params.ensemble_forward_batch_size,
                progress_bar=params.progress_bar,
                ) # (num_candidate_points, num_samples)
        if zero_gamma_model is not None:
            zero_gamma_pred_means, zero_gamma_pred_vars = train.ensemble_forward(
                    zero_gamma_model, 
                    X, 
                    params.ensemble_forward_batch_size
                    )
            zero_gamma_pred_means = zero_gamma_pred_means.detach()
            zero_gamma_pred_vars = zero_gamma_pred_vars.detach()

        ind_top_ood_pred_stats = bopt.get_ind_top_ood_pred_stats(
                pre_ack_pred_means,
                torch.sqrt(pre_ack_pred_vars),
                Y,
                train_Y_cur,
                params.output_noise_dist_fn, 
                set(),
                params.sigmoid_coeff,
                single_gaussian=params.single_gaussian_test_nll,
                num_to_eval=params.num_ind_to_eval,
                )
        print('ind_top_ood_pred_stats:', pprint.pformat(ind_top_ood_pred_stats))

        test_pred_stats = bopt.get_pred_stats(
                pre_ack_pred_means[:, test_idx],
                torch.sqrt(pre_ack_pred_vars[:, test_idx]),
                Y[test_idx],
                params.output_noise_dist_fn, 
                params.sigmoid_coeff,
                train_Y=train_Y_cur,
                )
        print('test_pred_stats:', pprint.pformat(test_pred_stats))

        if params.report_zero_gamma:
            if zero_gamma_model is not None:
                zero_gamma_test_pred_stats = bopt.get_pred_stats(
                        zero_gamma_pred_means[:, test_idx],
                        torch.sqrt(zero_gamma_pred_vars[:, test_idx]),
                        Y[test_idx],
                        params.output_noise_dist_fn, 
                        params.sigmoid_coeff,
                        train_Y=train_Y_cur,
                        )
                print('zero_gamma_test_pred_stats:', pprint.pformat(zero_gamma_test_pred_stats))
            else:
                zero_gamma_test_pred_stats = test_pred_stats

    ood_pred_stats = None
    if ood_inputs is not None:
        assert ood_labels is not None
        assert ood_inputs.shape[0] == ood_labels.shape[0]

        with torch.no_grad():
            ood_preds_means, ood_preds_vars = train.ensemble_forward(
                    init_model, 
                    ood_X, 
                    params.ensemble_forward_batch_size,
                    progress_bar=params.progress_bar,
                    ) # (num_candidate_points, num_samples)

            if zero_gamma_model is not None:
                zero_gamma_ood_preds_means, zero_gamma_ood_preds_vars = train.ensemble_forward(
                        zero_gamma_model, 
                        ood_X, 
                        params.ensemble_forward_batch_size
                        )
                zero_gamma_ood_preds_means = zero_gamma_ood_preds_means.detach()
                zero_gamma_ood_preds_vars = zero_gamma_ood_preds_vars.detach()

            ood_pred_stats = bopt.get_pred_stats(
                    ood_preds_means,
                    torch.sqrt(ood_preds_vars),
                    ood_Y,
                    params.output_noise_dist_fn, 
                    params.sigmoid_coeff,
                    train_Y=train_Y_cur,
                    )
            print('ood_pred_stats:', pprint.pformat(ood_pred_stats))
            if params.report_zero_gamma:
                if zero_gamma_model is not None:
                    zero_gamma_ood_pred_stats = bopt.get_pred_stats(
                            zero_gamma_ood_preds_means,
                            torch.sqrt(zero_gamma_ood_preds_vars),
                            ood_Y,
                            params.output_noise_dist_fn, 
                            params.sigmoid_coeff,
                            train_Y=train_Y_cur,
                            )
                    print('zero_gamma_ood_pred_stats:', pprint.pformat(zero_gamma_ood_pred_stats))
                else:
                    zero_gamma_ood_pred_stats = ood_pred_stats

        if logging is not None and not params.log_all_train_iter:
            logging[0] = None

        to_save_dict = {
            'model_state_dict': init_model.state_dict(), 
            'logging': logging,
            'best_gamma': best_gamma,
            'train_idx': torch.from_numpy(train_idx),
            'global_params': vars(params),
            'ind_top_ood_pred_stats': ind_top_ood_pred_stats,
            'ood_pred_stats': ood_pred_stats,
            'test_pred_stats': test_pred_stats,
            }
        if params.report_zero_gamma:
            to_save_dict['zero_gamma_test_pred_stats'] = zero_gamma_test_pred_stats
            to_save_dict['zero_gamma_ood_pred_stats'] = zero_gamma_ood_pred_stats

        torch.save(to_save_dict, init_model_path)
    else:
        to_save_dict = {
            'model_state_dict': init_model.state_dict(), 
            'logging': logging,
            'best_gamma': best_gamma,
            'train_idx': torch.from_numpy(train_idx),
            'global_params': vars(params),
            'ind_top_ood_pred_stats': ind_top_ood_pred_stats,
            }
        torch.save(to_save_dict, init_model_path)

    with open(file_output_dir + "/stats.txt", 'w' if params.clean else 'a', buffering=1) as main_f:
        if not loaded and logging is not None:
            main_f.write(pprint.pformat(logging[1]) + "\n")

        for ack_batch_size in [params.ack_batch_size]:
            print('doing batch', ack_batch_size)
            batch_output_dir = file_output_dir + "/" + str(ack_batch_size)
            try:
                os.mkdir(batch_output_dir)
            except OSError as e:
                pass

            if params.predict_ood:
                ood_pred_model = train.OodPredModel(train_X_init.shape[1], params.ood_pred_emb_size).to(params.device)
                all_pred_means = []
                all_pred_variances = []

            idx_at_each_iter = [train_idx.tolist()]

            train_X_cur = train_X_init.clone()

            cur_model = copy.deepcopy(init_model)

            skip_idx_cur = set(train_idx.tolist() + test_idx.tolist() + val_idx.tolist())
            train_idx_cur = set(train_idx.tolist())
            ack_all_cur = set()

            if os.path.exists(batch_output_dir + "/" + str(params.num_acks-1) + ".pth") and not params.clean:
                print('already done batch', ack_batch_size)
                continue

            torch.save(to_save_dict, batch_output_dir + "/0.pth")
            temp_min_rank = min([sorted_labels_idx_to_rank[idx] for idx in train_idx_cur])
            print('min_rank', temp_min_rank)

            with open(batch_output_dir + "/stats.txt", 'w', buffering=1) as f:
                ack_iter_info = {
                        'ucb_beta': params.ucb,
                        }
                for ack_iter in range(params.num_acks):
                    batch_ack_output_file = batch_output_dir + "/" + str(ack_iter+1) + ".pth"

                    # test stats computation
                    print('doing ack_iter', ack_iter)
                    print('task_name:', task_name[task_iter], '; measure:', params.ack_fun, '; output folder', params.output_dir)

                    if params.mode == "bayes_opt":
                        if "empirical_cond_" in params.ack_fun:
                            if params.ack_fun == "empirical_cond_pdts":
                                ack_idx_to_condense = bopt.get_pdts_idx(
                                        pre_ack_pred_means, 
                                        params.num_diversity*ack_batch_size, 
                                        density=False)
                            elif params.ack_fun == "empirical_cond_er":
                                er = pre_ack_pred_means.mean(dim=0).view(-1)
                                std = pre_ack_pred_means.std(dim=0).view(-1)
                                er += params.ucb * std
                                er[list(skip_idx_cur)] = er.min()
                                er_sortidx = torch.sort(er, descending=True)[1]

                                ack_idx_to_condense = []
                                condense_size = params.num_diversity*ack_batch_size
                                for i in range(er_sortidx.shape[0]):
                                    idx = er_sortidx[i].item()
                                    if idx in skip_idx_cur:
                                        continue
                                    ack_idx_to_condense += [idx]
                                    if len(ack_idx_to_condense) == condense_size:
                                        break

                            cur_ack_idx = bopt.get_empirical_condensation_ack3(
                                    params,
                                    X,
                                    Y,
                                    pre_ack_pred_means,
                                    cur_model,
                                    params.re_train_lr,
                                    params.re_train_l2,
                                    ack_idx_to_condense,
                                    skip_idx_cur,
                                    ack_batch_size,
                                    idx_to_monitor=set(range(X.shape[0])) if params.empirical_stat == 'mes' else None,
                                    er_values=er,
                                    seen_batch_size=params.re_train_batch_size,
                                    stat_fn=bopt.compute_maxdist_entropy if params.empirical_stat == 'mes' else lambda preds, info : preds.std(dim=0),
                                    val_nll_metric="val" in params.empirical_stat,
                                    )

                            if len(cur_ack_idx) < ack_batch_size:
                                for i in range(er.shape[0]):
                                    idx = er_sortidx[i].item()
                                    if idx not in cur_ack_idx:
                                        assert idx not in skip_idx_cur, idx
                                        cur_ack_idx += [idx]
                                    if len(cur_ack_idx) == ack_batch_size:
                                        break

                        elif params.ack_fun == "maxminembcorr":
                            cur_ack_idx = bopt.acquire_batch_maxminembcorr_info(
                                    params,
                                    pre_ack_pred_means,
                                    cur_model,
                                    X,
                                    ack_batch_size,
                                    skip_idx_cur,
                                    device=params.device,
                                    )
                        elif "info" in params.ack_fun and "grad" not in params.ack_fun:
                            cur_ack_idx = bopt.get_info_ack(
                                    params,
                                    pre_ack_pred_means,
                                    ack_batch_size,
                                    skip_idx_cur,
                                    labels,
                                    info_measure=params.info_measure,
                                    )
                            if "rand" in params.ack_fun and params.rand_diversity_dist == "uniform":
                                unseen_idx2 = list(unseen_idx)
                                random.shuffle(unseen_idx2)
                                num_rand = utils.get_num_rand(params.num_rand_diversity)
                                if num_rand >= 1:
                                    cur_ack_idx = cur_ack_idx[:-num_rand]
                                    for idx in unseen_idx2:
                                        if len(cur_ack_idx) >= ack_batch_size:
                                            break
                                        if idx in cur_ack_idx:
                                            continue
                                        cur_ack_idx += [idx]
                        elif "grad" in params.ack_fun:
                            if "pdts" in params.ack_fun:
                                point_locs, point_preds = bopt.optimize_model_input_pdts(
                                        params, 
                                        cur_model.input_shape(),
                                        cur_model,
                                        input_transform=lambda x : x.view(x.shape[0], -1),
                                        hsic_diversity_lambda=params.hsic_diversity_lambda,
                                        normalize_hsic=params.normalize_hsic,
                                        )
                            elif "ucb" in params.ack_fun:
                                point_locs, point_preds = bopt.optimize_model_input_ucb(
                                        params,
                                        cur_model.input_shape(),
                                        cur_model,
                                        num_points_to_optimize=params.num_models,
                                        diversity_lambda=params.hsic_diversity_lambda,
                                        beta=params.ucb,
                                        )
                                assert ops.is_finite(point_locs)
                                assert ops.is_finite(point_preds)
                            else:
                                assert False, params.ack_fun + " ack_fun not implemented"

                            if "rand" in params.ack_fun:
                                assert "info" not in params.ack_fun
                                assert params.rand_diversity_dist == 'uniform'
                                num_rand = utils.get_num_rand(params.num_rand_diversity)
                                if num_rand >= 1:
                                    x, _ = bb_fn.sample(num_rand)
                                    point_locs[:num_rand, :] = x
                            elif "info" in params.ack_fun:
                                if params.measure == "pes":
                                    hsic_batch, best_hsic = bopt.acquire_batch_via_grad_opt_location(
                                            params,
                                            cur_model,
                                            cur_model.input_shape(),
                                            point_locs, 
                                            ack_batch_size,
                                            device=params.device,
                                            hsic_condense_penalty=params.hsic_condense_penalty,
                                            input_transform=lambda x : x.view(x.shape[0], -1),
                                            normalize_hsic=params.normalize_hsic,
                                            min_hsic_increase=params.min_hsic_increase,
                                            )
                                elif params.measure == "mves":
                                    hsic_batch, best_hsic = bopt.acquire_batch_via_grad_hsic2(
                                            params, 
                                            cur_model,
                                            cur_model.input_shape(),
                                            opt_values,
                                            ack_batch_size,
                                            device=params.device,
                                            hsic_condense_penalty=params.hsic_condense_penalty,
                                            input_transform=lambda x : x.view(x.shape[0], -1),
                                            normalize_hsic=params.normalize_hsic,
                                            )

                                assert hsic_batch.shape[1:] == X.shape[1:], "%s[1:] == %s[1:]" % (hsic.batch.shape, X.shape)
                                assert hsic_batch.shape[0] <= ack_batch_size

                                temp_rand_idx = torch.randperm(point_locs.shape[0])
                                point_locs = point_locs[temp_rand_idx]
                                point_locs[:hsic_batch.shape[0], :] = hsic_batch

                            assert point_locs.shape[0] >= ack_batch_size
                            X2 = point_locs[:ack_batch_size]
                            Y2 = bb_fn(X2)

                            assert not ops.is_nan(Y2)
                            assert not ops.is_inf(Y2)

                            old_n = X.shape[0]

                            X = torch.cat([X, X2], dim=0)
                            Y = torch.cat([Y, Y2], dim=0)
                            inputs = np.concatenate([inputs, X2.cpu().numpy()], axis=0)
                            labels = np.concatenate([labels, Y2.cpu().numpy()], axis=0)

                            cur_ack_idx = [i+old_n for i in range(X2.shape[0])]
                            print('shapes', X2.shape, X.shape, Y.shape, Y2.shape, old_n, cur_ack_idx)
                        elif params.ack_fun == "kb":
                            cur_ack_idx = bopt.get_kriging_believer_ack(
                                    params,
                                    cur_model,
                                    [train_X_cur, train_Y_cur, X, Y],
                                    ack_batch_size,
                                    skip_idx_cur,
                                    train.train_ensemble,
                                    )
                        elif params.ack_fun == "bagging_er":
                            cur_ack_idx = bopt.get_bagging_er(
                                    params,
                                    cur_model,
                                    [train_X_cur, train_Y_cur, X, Y],
                                    ack_batch_size,
                                    skip_idx_cur,
                                    )
                        elif params.ack_fun == "empirical_kb":
                            cur_ack_idx = bopt.get_empirical_kriging_believer_ack(
                                    params,
                                    cur_model,
                                    [train_X_cur, train_Y_cur, X, Y],
                                    ack_batch_size,
                                    skip_idx_cur,
                                    train.train_ensemble,
                                    cur_rmse,
                                    )
                        elif params.ack_fun == "nb_mcts":
                            cur_ack_idx = bopt.get_nb_mcts_ack(
                                    params,
                                    cur_model,
                                    [train_X_cur, train_Y_cur, X, Y],
                                    ack_batch_size,
                                    skip_idx_cur,
                                    train.train_ensemble,
                                    normalize_fn=utils.sigmoid_standardization if params.sigmoid_coeff > 0 else utils.normal_standardization,
                                    )
                        elif "_ucb" in params.ack_fun:
                            normalize_fn=utils.sigmoid_standardization if params.sigmoid_coeff > 0 else utils.normal_standardization
                            mean_temp = train_Y_cur.mean()
                            std_temp = train_Y_cur.std()
                            train_Y_temp = normalize_fn(train_Y_cur, mean_temp, std_temp, exp=torch.exp)
                            cur_ack_idx = bopt.get_noninfo_ack(
                                    params,
                                    params.ack_fun,
                                    pre_ack_pred_means,
                                    ack_batch_size,
                                    skip_idx_cur,
                                    ack_iter_info=ack_iter_info,
                                    best_so_far=train_Y_temp.max(),
                                    )
                            if "rand" in params.ack_fun and params.rand_diversity_dist == "uniform":
                                unseen_idx2 = list(unseen_idx)
                                random.shuffle(unseen_idx2)
                                num_rand = utils.get_num_rand(params.num_rand_diversity)
                                if num_rand >= 1:
                                    cur_ack_idx = cur_ack_idx[:-num_rand]
                                    for idx in unseen_idx2:
                                        if len(cur_ack_idx) >= ack_batch_size:
                                            break
                                        if idx in cur_ack_idx:
                                            continue
                                        cur_ack_idx += [idx]
                        else:
                            assert False, params.ack_fun + " not implemented"
                    elif params.mode == "active_learning":
                        if params.ack_fun == "uniform":
                            cur_ack_idx = al.get_uniform_ack(
                                    params,
                                    pre_ack_pred_means.shape[1],
                                    ack_batch_size,
                                    skip_idx_cur,
                                    )
                        elif params.ack_fun == "max_std":
                            cur_ack_idx = al.get_max_std_ack(
                                    params,
                                    pre_ack_pred_means,
                                    ack_batch_size,
                                    skip_idx_cur,
                                    )
                        elif params.ack_fun == "max_var_ratio":
                            cur_ack_idx = al.get_max_var_ratio_ack(
                                    params,
                                    pre_ack_pred_means,
                                    pre_ack_pred_vars,
                                    ack_batch_size,
                                    skip_idx_cur,
                                    )
                        else:
                            assert False, params.ack_fun + " not implemented"

                    unseen_idx = set(range(Y.shape[0])).difference(skip_idx_cur.union(cur_ack_idx))
                    if params.ack_change_stat_logging and params.mode == "bayes_opt":
                        rand_idx, old_std, old_nll, corr_rand, hsic_rand, mi_rand, ack_batch_hsic, rand_batch_hsic = bopt.pairwise_logging_pre_ack(
                                params,
                                pre_ack_pred_means,
                                pre_ack_pred_vars,
                                skip_idx_cur,
                                cur_ack_idx,
                                unseen_idx,
                                Y,
                                do_hsic_rand=False,
                                do_mi_rand=False,
                                do_batch_hsic=params.ack_hsic_stat_logging,
                                )

                    if params.empirical_ack_change_stat_logging:
                        empirical_stat_fn = lambda preds, info : preds.std(dim=0)
                        idx_to_monitor = cur_ack_idx
                        #idx_to_monitor = np.random.choice(list(unseen_idx), 20, replace=False).tolist()
                        #idx_to_monitor = ack_idx_to_condense
                        guess_ack_stats, old_ack_stats = bopt.compute_ack_stat_change(
                            params,
                            X,
                            Y,
                            pre_ack_pred_means,
                            cur_ack_idx,
                            cur_model,
                            params.re_train_lr,
                            params.re_train_l2,
                            skip_idx_cur,
                            idx_to_monitor,
                            stat_fn=empirical_stat_fn,
                            seen_batch_size=params.re_train_batch_size,
                            )

                    assert type(cur_ack_idx) == list
                    ack_all_cur.update(cur_ack_idx)
                    skip_idx_cur.update(cur_ack_idx)
                    train_idx_cur.update(cur_ack_idx)
                    expected_num_points = (ack_iter+1)*ack_batch_size
                    idx_at_each_iter += [cur_ack_idx]

                    if params.predict_ood:
                        with torch.no_grad():
                            idx_flatten = [idx for idx_list in idx_at_each_iter for idx in idx_list]
                            all_pred_means += [pre_ack_pred_means[:, idx_flatten]]
                            all_pred_variances += [pre_ack_pred_vars[:, idx_flatten]]

                            #idx_to_predict_ood = cur_ack_idx
                            idx_to_predict_ood = range(X.shape[0])
                            ood_prediction = bopt.predict_ood(
                                    ood_pred_model,
                                    idx_flatten[:-ack_batch_size],
                                    idx_to_predict_ood,
                                    X,
                                    )
                            ood_prediction = ood_prediction.cpu().numpy()

                            if params.sigmoid_coeff > 0:
                                Y2 = utils.sigmoid_standardization(
                                        Y[idx_to_predict_ood],
                                        Y[idx_flatten[:-ack_batch_size]].mean(),
                                        Y[idx_flatten[:-ack_batch_size]].std(),
                                        exp=torch.exp)
                            else:
                                Y2 = utils.normal_standardization(
                                        Y[idx_to_predict_ood],
                                        Y[idx_flatten[:-ack_batch_size]].mean(),
                                        Y[idx_flatten[:-ack_batch_size]].std(),
                                        exp=torch.exp)

                            true_nll = cur_model.get_per_point_nll(
                                    Y2,
                                    pre_ack_pred_means[:, idx_to_predict_ood],
                                    pre_ack_pred_vars[:, idx_to_predict_ood],
                                    )
                            true_nll = true_nll.cpu().numpy()

                        nll_pred_corr = [
                                pearsonr(ood_prediction, true_nll)[0],
                                kendalltau(ood_prediction, true_nll)[0],
                                ]
                        print('nll_pred_corr pearson: %0.5f ; kt: %0.5f' % (nll_pred_corr[0], nll_pred_corr[1]))

                        ack_std = pre_ack_pred_means[:, idx_to_predict_ood].std(dim=0).cpu().numpy()
                        std_ack_pred_corr = [
                                pearsonr(ack_std, true_nll)[0],
                                kendalltau(ack_std, true_nll)[0],
                                ]
                        print('std_pred_corr pearson: %0.5f ; kt: %0.5f' % (std_ack_pred_corr[0], std_ack_pred_corr[1]))

                        ack_var = pre_ack_pred_means[:, idx_to_predict_ood].var(dim=0).cpu().numpy()
                        var_ack_pred_corr = [
                                pearsonr(ack_var, true_nll)[0],
                                kendalltau(ack_var, true_nll)[0],
                                ]
                        print('var_pred_corr pearson: %0.5f ; kt: %0.5f' % (var_ack_pred_corr[0], var_ack_pred_corr[1]))

                        bopt.train_ood_pred(
                                params,
                                ood_pred_model,
                                cur_model,
                                all_pred_means,
                                all_pred_variances,
                                X,
                                Y,
                                idx_at_each_iter,
                                )

                    #print("cur_ack_idx:", cur_ack_idx)
                    #print('er_labels', labels[cur_ack_idx])
                    #print('ack_labels', labels[cur_ack_idx].max(), labels[cur_ack_idx].min(), labels[cur_ack_idx].mean())

                    assert len(skip_idx_cur) == int(params.init_train_examples + len(test_idx) + len(val_idx)) + expected_num_points, str(len(skip_idx_cur)) + " == " + str(int(params.init_train_examples + len(test_idx) + len(val_idx)) + expected_num_points)

                    new_idx = list(train_idx_cur)
                    random.shuffle(new_idx)
                    new_idx = torch.LongTensor(new_idx)

                    train_X_cur = X[new_idx].clone().detach()
                    train_Y_cur = Y[new_idx].clone().detach()

                    print("train_X_cur.shape", train_X_cur.shape)
                    
                    assert train_X_cur.shape[0] == int(params.init_train_examples) + expected_num_points, str(train_X_cur.shape) + "[0] == " + str(int(params.init_train_examples) + expected_num_points)
                    assert train_Y_cur.shape[0] == train_X_cur.shape[0]

                    ensemble_init_rng, cur_model = train.reinit_model(
                        params,
                        untrained_model,
                        cur_model,
                        ensemble_init_rng
                        )

                    zero_gamma_model = None
                    if params.project in ['imdb', 'wiki']:
                        logging, best_gamma, data_split_rng, zero_logging, zero_gamma_model, cur_model = train.hyper_param_train(
                            params,
                            cur_model,
                            [train_X_cur, train_Y_cur, val_X, val_Y, X, Y],
                            "re",
                            params.gammas,
                            params.unseen_reg,
                            data_split_rng=data_split_rng,
                            predict_info_models=predict_info_models if params.predict_mi else None,
                            sample_uniform_fn=sample_uniform_fn,
                            normalize_fn=utils.sigmoid_standardization if params.sigmoid_coeff > 0 else utils.normal_standardization,
                            num_epoch_iters=None if params.fixed_num_epoch_iters == 0 else params.fixed_num_epoch_iters,
                            unseen_idx=unseen_idx,
                            mod_adversarial_test=params.mod_adversarial_test,
                            hyper_params=[['adv_epsilon', 'langevin_lr', 'langevin_num_iter', 'langevin_beta'], [params.adv_epsilon, params.langevin_lr, params.langevin_num_iter, params.langevin_beta]],
                            )
                    else:
                        logging, best_gamma, data_split_rng, zero_logging, zero_gamma_model, cur_model = train.hyper_param_train(
                            params,
                            cur_model,
                            [train_X_cur, train_Y_cur, val_X, val_Y, X, Y],
                            "re",
                            params.gammas,
                            params.unseen_reg,
                            data_split_rng=data_split_rng,
                            predict_info_models=predict_info_models if params.predict_mi else None,
                            sample_uniform_fn=sample_uniform_fn,
                            normalize_fn=utils.sigmoid_standardization if params.sigmoid_coeff > 0 else utils.normal_standardization,
                            num_epoch_iters=None if params.fixed_num_epoch_iters == 0 else params.fixed_num_epoch_iters,
                            unseen_idx=unseen_idx,
                            mod_adversarial_test=params.mod_adversarial_test,
                            hyper_params=[['adv_epsilon', 'langevin_lr', 'langevin_num_iter', 'langevin_beta'], [params.adv_epsilon, params.langevin_lr, params.langevin_num_iter, params.langevin_beta]],
                            )
                    torch.cuda.empty_cache()
                    #print('point 7:', nvidia_smi())

                    if logging is not None and logging[1] is not None and 'val' in logging[1] and 'rmse' in logging[1]['val']:
                        cur_rmse = logging[1]['val']['rmse']
                    else:
                        cur_rmse = None

                    with torch.no_grad():
                        cur_model.eval()
                        pre_ack_pred_means, pre_ack_pred_vars = train.ensemble_forward(cur_model, X, params.ensemble_forward_batch_size) # (num_candidate_points, num_samples)
                        pre_ack_pred_means = pre_ack_pred_means.detach()
                        pre_ack_pred_vars = pre_ack_pred_vars.detach()
                        assert pre_ack_pred_means.shape[1] == Y.shape[0], "%s[1] == %s[0]" % (str(pre_ack_pred_means.shape), str(Y.shape))

                        #preds_for_cur = torch.max(preds - train_Y_cur.max(), torch.tensor(0.).to(params.device))
                        #assert preds_for_cur.shape == preds.shape, str(preds_for_cur.shape)

                        ind_top_ood_pred_stats = bopt.get_ind_top_ood_pred_stats(
                                pre_ack_pred_means,
                                torch.sqrt(pre_ack_pred_vars),
                                Y,
                                train_Y_cur,
                                params.output_noise_dist_fn, 
                                skip_idx_cur,
                                params.sigmoid_coeff,
                                single_gaussian=params.single_gaussian_test_nll,
                                num_to_eval=params.num_ind_to_eval,
                                )


                        test_pred_stats = bopt.get_pred_stats(
                                pre_ack_pred_means[:, test_idx],
                                torch.sqrt(pre_ack_pred_vars[:, test_idx]),
                                Y[test_idx],
                                params.output_noise_dist_fn, 
                                params.sigmoid_coeff,
                                train_Y=train_Y_cur,
                                )
                        print('ind_top_ood_pred:', pprint.pformat(ind_top_ood_pred_stats))

                        ood_pred_stats = None
                        if ood_inputs is not None:
                            assert ood_labels is not None
                            assert ood_inputs.shape[0] == ood_labels.shape[0]
                            ood_preds_means, preds_vars = train.ensemble_forward(cur_model, ood_X, params.ensemble_forward_batch_size) # (num_candidate_points, num_samples)
                            ood_pred_stats = bopt.get_pred_stats(
                                    ood_preds_means,
                                    torch.sqrt(ood_preds_vars),
                                    ood_Y,
                                    params.output_noise_dist_fn, 
                                    params.sigmoid_coeff,
                                    train_Y=train_Y_cur,
                                    )
                            print('ood_pred_stats:', pprint.pformat(ood_pred_stats))

                    # ucb beta selection
                    if params.ucb_step >= 0.04 and params.mode == "bayes_opt":
                        ucb_beta_range = np.arange(0, params.ucb+0.01, params.ucb_step)
                        new_ucb_beta, best_kt = bopt.get_best_ucb_beta(
                                pre_ack_pred_means[:, cur_ack_idx],
                                Y[cur_ack_idx],
                                ucb_beta_range,
                                )
                        print('best ucb_beta kt:', best_kt)
                        if new_ucb_beta is not None:
                            ack_iter_info['ucb_beta'] = new_ucb_beta
                    print('new ucb_beta:', ack_iter_info['ucb_beta'])

                    if (params.ack_change_stat_logging or params.empirical_ack_change_stat_logging) and params.mode == "bayes_opt":
                        with torch.no_grad():
                            post_ack_preds, post_ack_pred_vars = cur_model(X) # (num_samples, num_points)
                            post_ack_preds = post_ack_preds.detach()
                            post_ack_pred_vars = post_ack_pred_vars.detach()

                            if params.empirical_ack_change_stat_logging:
                                true_ack_stats = empirical_stat_fn(post_ack_preds[:, idx_to_monitor], None)
                                true_ack_stats -= old_ack_stats

                                true_ack_stats = true_ack_stats.cpu()
                                guess_ack_stats = guess_ack_stats.cpu()
                                p1 = pearsonr(true_ack_stats, guess_ack_stats)[0]
                                kt = kendalltau(true_ack_stats, guess_ack_stats)[0]

                                print('empirical p1: %0.5f, k1: %0.5f' % (p1, kt))

                            if params.ack_change_stat_logging:
                                stats_pred = None
                                if params.predict_mi:
                                    stats_pred = predict_info_models(X[rand_idx], post_ack_preds[:, rand_idx], X[cur_ack_idx])

                                corr_stats = bopt.pairwise_logging_post_ack(
                                        params,
                                        post_ack_preds[:, rand_idx],
                                        post_ack_pred_vars[:, rand_idx],
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
                    ack_labels = labels[ack_array]

                    to_save_dict = {
                            'model_state_dict': cur_model.state_dict(), 
                            'logging': logging,
                            'best_gamma': best_gamma,
                            'ack_idx': torch.from_numpy(ack_array),
                            'ack_labels': torch.from_numpy(ack_labels),
                            'idx_at_each_iter': idx_at_each_iter,
                            'ack_iter_info': ack_iter_info,
                            'ind_top_ood_pred_stats': ind_top_ood_pred_stats,
                            'ood_pred_stats': ood_pred_stats,
                            'test_pred_stats': test_pred_stats,
                            'corr_stats': torch.tensor(corr_stats) if params.ack_change_stat_logging else None,
                            }

                    # inference regret computation using retrained ensemble
                    if params.mode == "bayes_opt":
                        if params.project == "test_fn":
                            ir = bopt.compute_ir_regret_ensemble_grad(
                                    params,
                                    cur_model,
                                    bb_fn,
                                    )
                            print ('ir regret:', ir)
                        else:
                            s, ir, ir_sortidx = bopt.compute_ir_regret_ensemble(
                                    params,
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

                        if logging is not None and not params.log_all_train_iter:
                            logging[0] = None

                        best_so_far_idx = ack_array[ack_labels.argmax()]
                        best_so_far_idx_rank = sorted_labels_idx_to_rank[best_so_far_idx]
                        if params.project == "test_fn":
                            ack_rel_opt_value = bb_fn.optimum()-ack_labels.max()
                            ir_rel_opt_value = min(ack_rel_opt_value, bb_fn.optimum()-ir)
                        else:
                            ack_rel_opt_value = idx_to_rel_opt_value[best_so_far_idx]
                            ir_rel_opt_value = ack_rel_opt_value
                            if idx_to_rel_opt_value[ir_sortidx[-1]] > ir_rel_opt_value:
                                ir_rel_opt_value = idx_to_rel_opt_value[ir_sortidx[-1]]

                        print('best so far:', labels[best_so_far_idx])
                        print('regret:', ack_rel_opt_value, ir_rel_opt_value)
                        print('rank regret:', best_so_far_idx_rank)

                        if params.project == "test_fn":
                            temp = {
                                'ack_rel_opt_value': ack_rel_opt_value,
                                'ir_rel_opt_value': ir_rel_opt_value,
                                'rank_regret': best_so_far_idx_rank,
                                }
                        else:
                            temp = {
                                'ack_rel_opt_value': ack_rel_opt_value,
                                'ir_batch_cur': torch.from_numpy(ir),
                                'ir_batch_cur_idx': torch.from_numpy(ir_sortidx),
                                'ir_rel_opt_value': ir_rel_opt_value,
                                'idx_frac': torch.tensor(idx_frac),
                                'rank_regret': best_so_far_idx_rank,
                                }
                        for k in temp:
                            to_save_dict[k] = temp[k]
                    elif params.mode == "active_learning":
                        pass
                    else:
                        assert False, params.mode + " not implemented"

                    torch.save(to_save_dict, batch_ack_output_file)
                    sys.stdout.flush()
                    
            #main_f.write(str(ack_batch_size) + "\t" + s + "\n")

sys.stdout.close()
