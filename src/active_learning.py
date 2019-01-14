
import os
import random
import copy
import gc
import sys
import numpy as np
import math
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.stats import kendalltau, pearsonr
import torch
from tqdm import tnrange, trange
import torch.nn as nn

def get_max_std_ack(
    params,
    pre_ack_pred_means,
    ack_batch_size,
    skip_idx,
):
    skip_idx = list(skip_idx)
    std = pre_ack_pred_means.std(dim=0)
    std[skip_idx] = std.min()
    std_sort, std_sort_idx = torch.sort(std, descending=True)

    return std_sort_idx[:ack_batch_size].cpu().numpy().tolist()


def get_uniform_ack(
    params,
    N,
    ack_batch_size,
    skip_idx,
):
    unseen_idx = list(set(range(N)).difference(set(skip_idx)))
    ack_batch = np.random.choice(unseen_idx, size=ack_batch_size, replace=False).tolist()

    return ack_batch
