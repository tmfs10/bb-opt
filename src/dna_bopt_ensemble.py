
import sys
sys.path.append('/cluster/sj1')
sys.path.append('/cluster/sj1/bb_opt/src')

import os
import torch
import pprint
import random
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.distributions as tdist
import numpy as np
from scipy.stats import kendalltau, pearsonr
import dna_bopt as dbopt
import bayesian_opt as bopt
import active_learning as al
from gpu_utils.utils import gpu_init
import pandas as pd
import copy
import non_matplotlib_utils as utils
import datetime
import parsing
import argparse
import ops
from deep_ensemble_sid import NNEnsemble

parser = argparse.ArgumentParser()
parsing.add_parse_args(parser)
parsing.add_parse_args_nongrad(parser)
parsing.add_parse_args_ensemble(parser)
parsing.add_parse_args_wrongness(parser)
parsing.add_parse_imdbwiki_args(parser)

params = parsing.parse_args(parser)

print('PARAMS:')
for k, v in vars(params).items():
    print(k, v)
do_model_hparam_search = len(params.gammas) > 1

gpu_id = gpu_init(best_gpu_metric="mem")
print(f"Running on GPU {gpu_id}")
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
params.device = device

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
    filenames = [k.strip() for k in open(params.filename_file).readlines()][:params.num_test_tfs]
    task_name += [filename]
    sample_uniform_fn = dbopt.dna_sample_uniform
elif params.project in ['imdb', 'wiki']:
    task_name += [params.project]
    sample_uniform_fn = dbopt.image_sample_uniform

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

    if params.project == 'dna_binding':
        inputs = np.load(params.data_dir + "/" + task_name[task_iter] + "/inputs.npy").astype(np.float32)
        labels = np.load(params.data_dir + "/" + task_name[task_iter] + "/labels.npy").astype(np.float32)
        if params.take_log:
            labels = np.log(labels)
    elif params.project in ['imdb', 'wiki']:
        inputs, labels, gender = utils.load_data_wiki_sid(
                params.data_dir,
                params.project,
                )
        male_gender = (gender == 1)
        female_gender = (gender == 0)
        
        ood_inputs = inputs[female_gender].astype(np.float32)
        ood_labels = labels[female_gender].astype(np.float32)

        if params.num_ood_to_eval > 0:
            rand_idx = np.random.choice(ood_inputs.shape[0], size=(params.num_ood_to_eval,), replace=False).tolist()
            ood_inputs = ood_inputs[rand_idx]
            ood_labels = ood_labels[rand_idx]

        inputs = inputs[male_gender].astype(np.float32)
        labels = labels[male_gender].astype(np.float32)

        #temp_idx = np.random.choice(inputs.shape[0], size=(2000,), replace=False).tolist()
        #inputs = inputs[temp_idx]
        #labels = labels[temp_idx]

        X = torch.tensor(inputs, device=device)
        Y = torch.tensor(labels, device=device)

        ood_X = torch.tensor(ood_inputs, device=device)
        ood_Y = torch.tensor(ood_labels, device=device)

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
    train_idx, _, test_idx, data_split_rng = utils.train_val_test_split(
            indices, 
            split=[params.init_train_examples, 0],
            rng=data_split_rng,
            )

    train_inputs = inputs[train_idx]
    train_labels = labels[train_idx]

    test_idx = test_idx[:params.num_test_points]
    test_inputs = inputs[test_idx]
    test_labels = inputs[test_idx]

    top_frac_idx = [set(labels_sort_idx[-int(labels.shape[0]*per):].tolist()) for per in [0.01, 0.05, 0.1, 0.2]]

    print('label stats:', labels.mean(), labels.max(), labels.std())
    print('num_test_idx:', len(test_idx))

    train_X_init = torch.FloatTensor(train_inputs).to(device)
    train_Y_cur = torch.FloatTensor(train_labels).to(device)

    cur_rng = ops.get_rng_state()
    ops.set_rng_state(ensemble_init_rng)

    if params.project == "dna_binding":
        init_model = dbopt.get_model_nn_ensemble(
                inputs.shape[1], 
                params.num_models, 
                params.num_hidden, 
                sigmoid_coeff=params.sigmoid_coeff, 
                device=params.device,
                separate_mean_var=params.separate_mean_var,
                )
    elif params.project in ["wiki", "imdb"]:
        init_model = NNEnsemble.get_model_resnet(
                inputs.shape[1], 
                params.num_models, 
                params.resnet_depth,
                params.resnet_width_factor,
                params.resnet_dropout,
                device=params.device,
                )

    ensemble_init_rng = ops.get_rng_state()
    ops.set_rng_state(cur_rng)

    predict_info_models = None
    if params.predict_mi:
        predict_kernel_dim = 10
        predict_info_models = dbopt.PredictInfoModels(
                inputs.shape[1],
                params.ack_emb_kernel_dim,
                params.num_models,
                predict_kernel_dim,
                params.predict_stddev,
                params.predict_mmd,
                params.predict_nll,
                ).to(params.device)

    init_model_path = file_output_dir + "/init_model.pth"
    loaded = False

    with torch.no_grad():
        preds, preds_vars = dbopt.ensemble_forward(init_model, X, params.re_train_batch_size) # (num_candidate_points, num_samples)
        ind_top_ood_pred_stats = bopt.get_ind_top_ood_pred_stats(
                preds,
                torch.sqrt(preds_vars),
                Y,
                train_Y_cur,
                params.output_noise_dist_fn, 
                set(),
                params.sigmoid_coeff,
                single_gaussian=params.single_gaussian_test_nll,
                num_to_eval=params.num_ind_to_eval,
                )

        test_pred_stats = bopt.get_pred_stats(
                preds[:, test_idx],
                torch.sqrt(preds_vars[:, test_idx]),
                Y[test_idx],
                params.output_noise_dist_fn, 
                params.sigmoid_coeff,
                train_Y=train_Y_cur,
                )
        print('test_pred_stats:', pprint.pformat(test_pred_stats))

    ood_pred_stats = None
    if ood_inputs is not None:
        assert ood_labels is not None
        assert ood_inputs.shape[0] == ood_labels.shape[0]

        with torch.no_grad():
            ood_preds_means, ood_preds_vars = dbopt.ensemble_forward(init_model, ood_X, params.init_train_batch_size) # (num_candidate_points, num_samples)
            ood_pred_stats = bopt.get_pred_stats(
                    ood_preds_means,
                    torch.sqrt(ood_preds_vars),
                    ood_Y,
                    params.output_noise_dist_fn, 
                    params.sigmoid_coeff,
                    train_Y=train_Y_cur,
                    )
            print('ood_pred_stats:', pprint.pformat(ood_pred_stats))

    if os.path.isfile(init_model_path) and not params.clean:
        loaded = True
        checkpoint = torch.load(init_model_path)
        init_model.load_state_dict(checkpoint['model_state_dict'])
        logging = checkpoint["logging"]
        if "train_idx" in checkpoint:
            train_idx = checkpoint["train_idx"].numpy()
    else:

        init_model, logging, best_gamma, data_split_rng = dbopt.hyper_param_train(
            params,
            init_model,
            [train_X_init, train_Y_cur, X, Y],
            stage="init",
            data_split_rng=data_split_rng,
            predict_info_models=predict_info_models if params.predict_mi else None,
            sample_uniform_fn=sample_uniform_fn,
            )

        cur_rmse = logging[1]['val']['rmse']

        if not params.log_all_train_iter:
            logging[0] = None
        torch.save({
            'model_state_dict': init_model.state_dict(), 
            'logging': logging,
            'best_gamma': best_gamma,
            'train_idx': torch.from_numpy(train_idx),
            'global_params': vars(params),
            'ind_top_ood_pred_stats': ind_top_ood_pred_stats,
            'ood_pred_stats': ood_pred_stats,
            'test_pred_stats': test_pred_stats,
            }, init_model_path)

    with open(file_output_dir + "/stats.txt", 'w' if params.clean else 'a', buffering=1) as main_f:
        if not loaded:
            main_f.write(pprint.pformat(logging[1]) + "\n")

        for ack_batch_size in [params.ack_batch_size]:
            print('doing batch', ack_batch_size)
            batch_output_dir = file_output_dir + "/" + str(ack_batch_size)
            try:
                os.mkdir(batch_output_dir)
            except OSError as e:
                pass

            if params.predict_ood:
                ood_pred_model = dbopt.OodPredModel(train_X_init.shape[1], params.ood_pred_emb_size).to(params.device)
                all_pred_means = []
                all_pred_variances = []

            idx_at_each_iter = [train_idx.tolist()]

            train_X_cur = train_X_init.clone()
            train_Y_cur = train_Y_cur.clone()

            cur_model = copy.deepcopy(init_model)

            skip_idx_cur = set(train_idx.tolist())
            ack_all_cur = set()

            if os.path.exists(batch_output_dir + "/" + str(params.num_acks-1) + ".pth") and not params.clean:
                print('already done batch', ack_batch_size)
                continue

            with open(batch_output_dir + "/stats.txt", 'w', buffering=1) as f:
                ack_iter_info = {
                        'ucb_beta': params.ucb,
                        }
                for ack_iter in range(params.num_acks):
                    batch_ack_output_file = batch_output_dir + "/" + str(ack_iter) + ".pth"

                    # test stats computation
                    print('doing ack_iter', ack_iter)
                    with torch.no_grad():
                        pre_ack_pred_means, pre_ack_pred_vars = dbopt.ensemble_forward(cur_model, X, params.re_train_batch_size) # (num_candidate_points, num_samples)
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
                        print('test_pred_stats:', pprint.pformat(test_pred_stats))

                        ood_pred_stats = None
                        if ood_inputs is not None:
                            assert ood_labels is not None
                            assert ood_inputs.shape[0] == ood_labels.shape[0]
                            ood_preds_means, preds_vars = dbopt.ensemble_forward(cur_model, ood_X, params.init_train_batch_size) # (num_candidate_points, num_samples)
                            ood_pred_stats = bopt.get_pred_stats(
                                    ood_preds_means,
                                    torch.sqrt(ood_preds_vars),
                                    ood_Y,
                                    params.output_noise_dist_fn, 
                                    params.sigmoid_coeff,
                                    train_Y=train_Y_cur,
                                    )
                            print('ood_pred_stats:', pprint.pformat(ood_pred_stats))

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

                        elif params.ack_fun == "info":
                            cur_ack_idx = bopt.get_info_ack(
                                    params,
                                    pre_ack_pred_means,
                                    ack_batch_size,
                                    skip_idx_cur,
                                    labels,
                                    )
                        elif params.ack_fun == "kb":
                            cur_ack_idx = bopt.get_kriging_believer_ack(
                                    params,
                                    cur_model,
                                    [train_X_cur, train_Y_cur, X, Y],
                                    ack_batch_size,
                                    skip_idx_cur,
                                    dbopt.train_ensemble,
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
                                    dbopt.train_ensemble,
                                    cur_rmse,
                                    )
                        elif "_ucb" in params.ack_fun:
                            cur_ack_idx = bopt.get_noninfo_ack(
                                    params,
                                    params.ack_fun,
                                    pre_ack_pred_means,
                                    ack_batch_size,
                                    skip_idx_cur,
                                    ack_iter_info=ack_iter_info,
                                    )
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

                    assert len(skip_idx_cur) == int(params.init_train_examples) + expected_num_points, str(len(skip_idx_cur)) + " == " + str(int(params.init_train_examples) + expected_num_points)

                    new_idx = list(skip_idx_cur)
                    random.shuffle(new_idx)
                    new_idx = torch.LongTensor(new_idx)

                    train_X_cur = X.new_tensor(X[new_idx])
                    train_Y_cur = Y.new_tensor(Y[new_idx])

                    print("train_X_cur.shape", train_X_cur.shape)
                    
                    assert train_X_cur.shape[0] == int(params.init_train_examples) + expected_num_points, str(train_X_cur.shape) + "[0] == " + str(int(params.init_train_examples) + expected_num_points)
                    assert train_Y_cur.shape[0] == train_X_cur.shape[0]

                    cur_model, ensemble_init_rng = dbopt.reinit_model(
                        params,
                        init_model,
                        cur_model,
                        ensemble_init_rng
                        )
                    cur_model, logging, best_gamma, data_split_rng = dbopt.hyper_param_train(
                        params,
                        cur_model,
                        [train_X_cur, train_Y_cur, X, Y],
                        stage="re",
                        data_split_rng=data_split_rng,
                        predict_info_models=predict_info_models if params.predict_mi else None,
                        sample_uniform_fn=sample_uniform_fn,
                        )

                    cur_rmse = logging[1]['val']['rmse']

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
                        s, ir, ir_sortidx = bopt.compute_ir_regret_ensemble(
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

                        if not params.log_all_train_iter:
                            logging[0] = None

                        best_so_far_idx = ack_array[ack_labels.argmax()]
                        ack_rel_opt_value = idx_to_rel_opt_value[best_so_far_idx]
                        ir_rel_opt_value = ack_rel_opt_value
                        if idx_to_rel_opt_value[ir_sortidx[-1]] > ir_rel_opt_value:
                            ir_rel_opt_value = idx_to_rel_opt_value[ir_sortidx[-1]]

                        print('best so far:', labels[best_so_far_idx])
                        temp = {
                            'ack_rel_opt_value': ack_rel_opt_value,
                            'ir_batch_cur': torch.from_numpy(ir),
                            'ir_batch_cur_idx': torch.from_numpy(ir_sortidx),
                            'ir_rel_opt_value': ir_rel_opt_value,
                            'idx_frac': torch.tensor(idx_frac),
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
