
import numpy as np
import os
import pickle
from typing import Dict, Tuple, Sequence, Union, Callable, Optional
from rdkit import Chem
from rdkit.Chem import Descriptors, QED
import bb_opt.src.sascorer as sascorer
import pyro
import torch
import sys
import ops

def sigmoid(x, exp=np.exp):
  return 1.0 / (1.0 + exp(-x))

def get_early_stopping(
    max_patience: int, threshold: float = 1.0
) -> Callable[[float], bool]:
    old_min = float("inf")
    patience = max_patience

    def early_stopping(metric: float) -> bool:
        nonlocal old_min
        nonlocal patience
        nonlocal max_patience
        if metric < threshold * old_min:
            old_min = metric
            patience = max_patience
        else:
            patience -= 1

        return patience == 0

    return early_stopping

def logp(smiles_mol, add_hs=False):
    m = Chem.MolFromSmiles(smiles_mol)
    if m is None:
        return -100
    return Descriptors.MolLogP(m, add_hs)


def qed(smiles_mol):
    m = Chem.MolFromSmiles(smiles_mol)
    if m is None:
        return -100
    return QED.qed(m)


def sas(smiles_mol):
    m = Chem.MolFromSmiles(smiles_mol)
    if m is None:
        return -100
    return sascorer.calculateScore(m)


def all_scores(smiles_mol, add_hs=False):
    m = Chem.MolFromSmiles(smiles_mol)
    if m is None:
        return [-100, -100, -100]
    return [Descriptors.MolLogP(m, add_hs), QED.qed(m), sascorer.calculateScore(m)]


def save_pyro_model(base_path: str, optimizer):
    pyro.get_param_store().save(f"{base_path}.params")
    optimizer.save(f"{base_path}.opt")


def load_pyro_model(base_path: str, optimizer):
    pyro.get_param_store().load(f"{base_path}.params")
    optimizer.load(f"{base_path}.opt")


def train_val_test_split(n, split, shuffle=True, rng=None):
    if type(n) == int:
        idx = np.arange(n)
    else:
        idx = n

    if rng is not None:
        cur_rng = ops.get_rng_state()
        ops.set_rng_state(rng)

    if shuffle:
        np.random.shuffle(idx)

    if rng is not None:
        rng = ops.get_rng_state()
        ops.set_rng_state(cur_rng)

    if split[0] < 1:
        assert sum(split) <= 1.
        train_end = int(n * split[0])
        val_end = train_end + int(n * split[1])
    else:
        train_end = split[0]
        val_end = train_end + split[1]

    return idx[:train_end], idx[train_end:val_end], idx[val_end:], rng


def load_checkpoint(fname: str, model, optimizer: Optional = None) -> None:
    checkpoint = torch.load(fname)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state"])

def jointplot(
    x, y, axis_labels: Tuple[str, str] = ("Predicted", "True"), title: str = ""
):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()

    if isinstance(y, torch.Tensor):
        y = y.detach().cpu().numpy()

    ax = sns.jointplot(x, y, s=3, alpha=0.5)
    ax.set_axis_labels(*axis_labels)
    ax.ax_marg_x.set_title(title)
    return ax


def collated_expand(X, num_samples):
    X = X.unsqueeze(1)
    X = X.repeat([1] + [num_samples] + [1] * (len(X.shape) - 2)).view(
        [-1] + list(X.shape[2:])
    )
    return X

def print_progress(iteration, total, prefix='', suffix='', decimals=1, bar_length=100):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        bar_length  - Optional  : character length of bar (Int)
    """
    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = '█' * filled_length + '-' * (bar_length - filled_length)

    sys.stdout.write('%s |%s| %s%s\n' % (prefix, bar, percents, '%'))
    sys.stdout.write(suffix + "\n")
    sys.stdout.write('\033[F')
    sys.stdout.write('\033[F')
    sys.stdout.write('\033[F')
    sys.stdout.flush()

    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()

def get_path(*path_segments):
    return os.path.realpath(os.path.join(*[str(segment) for segment in path_segments]))

def save_checkpoint(fname: str, model, optimizer: Optional = None) -> None:
    os.makedirs(os.path.dirname(fname), exist_ok=True)

    checkpoint = {"model_state": model.state_dict()}
    if optimizer:
        checkpoint["optimizer_state"] = optimizer.state_dict()

    torch.save(checkpoint, fname)

def sigmoid_standardization(labels, mean, std, exp=np.exp):
    labels = sigmoid((labels - mean) / std, exp)
    return labels

def normal_standardization(labels, mean, std, exp=np.exp):
    labels = (labels - mean) / std
    return labels
