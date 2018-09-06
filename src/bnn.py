
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import pyro
import pyro.optim
import pickle
from typing import Union, Tuple, Optional, Callable, Dict, Set, List
from tqdm import tnrange
from bb_opt.src.bayesian_opt import (
    normal_priors, normal_variationals,
    spike_slab_priors, SpikeSlabNormal,
    make_bnn_model, make_guide,
    train, bnn_predict, optimize,
    get_model_bnn, acquire_batch_bnn_greedy, train_model_bnn,
    get_model_nn, acquire_batch_nn_greedy, train_model_nn
)

parser = argparse.ArgumentParser()
parser.add_argument('--save', type=str,  default='', help='path to save the final model')
random_label = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
params.add_parse_args(parser)
args = parser.parse_args()
args.save = "outputs/" + args.save+'.'+args.model + "." + random_label + ".pt"

np.random.seed(args.seed)
torch.manual_seed(args.seed)
