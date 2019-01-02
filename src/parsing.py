

import argparse
import torch.distributions as tdist
import yaml

def str2bool(v):
    if v.lower() in ('true', '1', 'y', 'yes'):
        return True
    elif v.lower() in ('false', '0', 'n', 'no'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected. Got' + v)

def strlower(v):
    return v.lower()

def float01(v):
    v = float(v)
    if v < 0 or v > 1:
        raise argparse.ArgumentTypeError('Value between 0 and 1 expected. Got' + float(v))
    return v

def str2dist(v):
    return getattr(tdist, v)

def atleast0int(v):
    v = int(v)
    assert v >= 0
    return v

def verify_model_update_mode(mode):
    mode = mode.lower().strip()
    assert mode in [
            "init_init",
            "new_init",
            "finetune",
            ]
    return mode


def verify_choose_type(choose_type):
    choose_type = [k.lower() for k in choose_type.split(',')]
    #assert len(choose_type) == 3, choose_type
    assert choose_type[0] in ["val", "train", "last"], choose_type
    assert choose_type[1] in ["nll", "kt_corr", "classify", "bopt"], choose_type
    assert choose_type[2] in ["ind", "ood"], choose_type
    return choose_type


def verify_empirical_stat(stat):
    stat = stat.lower().strip()
    assert stat in ["val_nll", "val_classify", "mes", "std"]
    return stat

def add_parse_args(parser):
    parser.add_argument('--config_file', type=str, nargs='+')
    parser.add_argument('--seed', type=int, 
            help='random seed')
    parser.add_argument('--model_init_seed', type=int, 
            help='random seed')
    parser.add_argument('--data_split_seed', type=int, 
            help='random seed')
    parser.add_argument('--device', type=str, 
            help='use cuda (default: True)')
    parser.add_argument('--clean', type=str2bool, 
            help='remove existing saved dir')
    parser.add_argument('--log_all_train_iter', type=str2bool)
    parser.add_argument("--ack_model_init_mode", type=verify_model_update_mode)
    parser.add_argument("--ack_change_stat_logging", type=str2bool)
    parser.add_argument("--ack_hsic_stat_logging", type=str2bool)
    parser.add_argument("--empirical_ack_change_stat_logging", type=str2bool)

    # train params
    parser.add_argument('--exclude_top', type=float01)
    parser.add_argument('--init_train_examples', type=int)
    parser.add_argument('--init_train_num_epochs', type=int)
    parser.add_argument('--init_train_lr', type=float)
    parser.add_argument('--init_train_batch_size', type=int)
    parser.add_argument('--re_train_num_epochs', type=int)
    parser.add_argument('--re_train_lr', type=float)
    parser.add_argument('--re_train_batch_size', type=int)
    parser.add_argument('--hyper_search_choose_type', type=verify_choose_type)
    parser.add_argument('--final_train_choose_type', type=verify_choose_type)
    parser.add_argument('--early_stopping', type=int, 
            help="num early stopping iters. 0 means no early stoppping")
    parser.add_argument('--val_frac', type=float,
            help="val frac to hold out as in-distribution validation set")
    parser.add_argument('--ood_val_frac', type=float,
            help="top frac to hold out as out-of-distribution validation set")
    parser.add_argument('--num_train_val_splits', type=int)
    parser.add_argument("--combine_train_val", type=str2bool)
    parser.add_argument("--gamma_cutoff", type=str2bool,
            help="Search starting from first gamma and end search as soon as new gamma is not best so far")
    parser.add_argument('--single_gaussian_test_nll', type=str2bool)
    parser.add_argument('--report_metric_train_std', type=str2bool)
    parser.add_argument('--empirical_stat', type=verify_empirical_stat)
    parser.add_argument('--empirical_stat_val_fraction', type=float)

    # model params
    parser.add_argument('--num_hidden', type=int)
    parser.add_argument('--sigmoid_coeff', type=float)
    parser.add_argument('--separate_mean_var', type=str2bool)
    parser.add_argument('--output_dist_fn', type=str2dist)
    parser.add_argument('--init_train_l2', type=float)
    parser.add_argument('--re_train_l2', type=float)

    parser.add_argument('--unseen_reg', type=strlower)
    parser.add_argument('--gammas', type=float, nargs='+',
            help="maxvar/defmean penalty")
    parser.add_argument('--ood_data_batch_factor', type=float)
    parser.add_argument('--take_log', type=str2bool)

    # data/output files/folders
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--filename_file', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--suffix', type=str)
    parser.add_argument('--num_test_tfs', type=int)

    # bopt params
    parser.add_argument('--ack_batch_size', type=int)
    parser.add_argument('--num_acks', type=int)

    # predictor params
    parser.add_argument('--predict_mi', type=str2bool)
    parser.add_argument('--predict_stddev', type=str2bool)
    parser.add_argument('--predict_nll', type=str2bool)
    parser.add_argument('--predict_mmd', type=str2bool)
    parser.add_argument('--num_predict_sample_points', type=int)
    parser.add_argument('--ack_emb_kernel_dim', type=int)


def add_parse_args_nongrad(parser):
    parser.add_argument('--ack_fun', type=strlower, 
            help="none/hsic/detk/pdts_ucb/var")
    parser.add_argument('--ucb', type=float, help="stddev coeff")
    parser.add_argument('--ucb_step', type=float)

    parser.add_argument('--num_diversity', type=int, 
            help='num top max-value dists/top ei sorted point dists')
    parser.add_argument('--diversity_coeff', type=float)

    parser.add_argument('--mves_greedy', type=str2bool, 
            help="use only first HSIC ordering, not sequential")
    parser.add_argument('--compare_w_old', type=str2bool, help="Build batch with replacement")
    parser.add_argument('--pred_weighting', type=int, 
            help="weight hsic; 1 - multiply by batch ei, 2 - add batch ei and multiply batch ei std")
    parser.add_argument('--divide_by_std', type=str2bool, 
            help="divide normalized hsic by hsic stddev")
    parser.add_argument('--measure', type=strlower, 
            help="mves/ei_mves_mix/ei_condense/ei_pdts_mix/cma_es")
    parser.add_argument('--mves_compute_batch_size', type=int)
    parser.add_argument('--hsic_kernel_fn', type=str)
    parser.add_argument('--min_hsic_increase', type=float, help="minimum hsic increase after which batch filled using ei")
    parser.add_argument('--normalize_hsic', type=str2bool)


def add_parse_args_grad(parser):
    parser.add_argument('--input_opt_lr', type=float)
    parser.add_argument('--input_opt_num_iter', type=int)
    parser.add_argument('--hsic_opt_lr', type=float)
    parser.add_argument('--hsic_opt_num_iter', type=int)
    parser.add_argument('--normalize_hsic', type=str2bool)
    parser.add_argument('--ack_num_model_samples', type=int)
    parser.add_argument('--measure', type=strlower)
    parser.add_argument('--hsic_kernel_fn', type=str)
    parser.add_argument('--hsic_diversity_lambda', type=float)
    parser.add_argument('--sparse_hsic_penalty', type=float)
    parser.add_argument('--sparse_hsic_threshold', type=float)
    parser.add_argument('--hsic_condense_penalty', type=float, nargs=2)


def add_parse_args_ensemble(parser):
    parser.add_argument('--num_models', type=int, help='number of models in ensemble')
    parser.add_argument('--adv_epsilon', type=float, help='adversarial epsilon')

def parse_args(parser):
    args = parser.parse_args()
    if len(args.config_file) > 0:
        for filename in args.config_file:
            with open(filename, 'r') as f:
                args2, leftovers = parser.parse_known_args(f.read().split())
                args_dict = vars(args)
                for k, v in vars(args2).items():
                    if args_dict[k] is not None:
                        continue
                    args_dict[k] = v

    if args.unseen_reg == "normal":
        args.gammas = [0.0]
    return args
