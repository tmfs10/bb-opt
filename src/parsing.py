

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

def add_parse_args(parser):
    parser.add_argument('--config_file', type=str, nargs='+')
    parser.add_argument('--seed', type=int, 
            help='random seed')
    parser.add_argument('--device', type=str, 
            help='use cuda (default: True)')
    parser.add_argument('--clean', type=str2bool, 
            help='remove existing saved dir')

    # train params
    parser.add_argument('--exclude_top', type=float01)
    parser.add_argument('--init_train_examples', type=int)
    parser.add_argument('--init_train_epochs', type=int)
    parser.add_argument('--train_lr', type=float)
    parser.add_argument('--train_batch_size', type=int)
    parser.add_argument('--retrain_num_epochs', type=int)
    parser.add_argument('--retrain_lr', type=float)
    parser.add_argument('--retrain_batch_size', type=int)
    parser.add_argument('--choose_type', type=strlower, help="last/val")

    # model params
    parser.add_argument('--num_hidden', type=int)
    parser.add_argument('--sigmoid_coeff', type=float)
    parser.add_argument('--output_dist_fn', type=str2dist)
    parser.add_argument('--train_l2', type=float)
    parser.add_argument('--retrain_l2', type=float)

    parser.add_argument('--unseen_reg', type=str)
    parser.add_argument('--gamma', type=float, 
            help="maxvar/defmean penalty")

    # data/output files/folders
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--filename_file', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--suffix', type=str)
    parser.add_argument('--num_test_tfs', type=int)

    # bopt params
    parser.add_argument('--ack_batch_size', type=int)
    parser.add_argument('--num_acks', type=int)

    # log params
    parser.add_argument('--single_gaussian_test_nll', type=str2bool)


def add_parse_args_ei(parser):
    parser.add_argument('--ei_diversity_measure', type=strlower, 
            help="none/hsic/detk/pdts_ucb/var")
    parser.add_argument('--ucb', type=float, help="stddev coeff")


def add_parse_args_mves(parser):
    parser.add_argument('--num_diversity', type=int, 
            help='num top max-value dists/top ei sorted point dists')

    parser.add_argument('--mves_greedy', type=str2bool, 
            help="use only first HSIC ordering, not sequential")
    parser.add_argument('--compare_w_old', type=str2bool, help="Build batch with replacement")
    parser.add_argument('--pred_weighting', type=int, 
            help="weight hsic; 1 - multiply by batch ei, 2 - add batch ei and multiply batch ei std")
    parser.add_argument('--divide_by_std', type=str2bool, 
            help="divide normalized hsic by hsic stddev")
    parser.add_argument('--measure', type=strlower, 
            help="mves/ei_mves_mix/ei_condense/ei_pdts_mix/cma_es")


def add_parse_args_ensemble(parser):
    parser.add_argument('--num_models', type=int, help='number of models in ensemble')

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

    return args
