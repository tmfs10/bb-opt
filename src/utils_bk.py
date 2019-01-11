import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import seaborn as sns
import numpy as np
import os
from os.path import join
import pickle
from typing import Dict, Tuple, Sequence, Union, Callable, Optional
from collections import namedtuple
from rdkit import Chem
from rdkit.Chem import Descriptors, QED
import bb_opt.src.sascorer as sascorer
import pyro
import torch
from sklearn.model_selection import train_test_split
from scipy.io import loadmat

from bb_opt.src import non_matplotlib_utils # utils giving Sid segfault cuz of matplotlib import when used from command-line

_Input_Labels = namedtuple("Input_Labels", ["inputs", "labels"])
_Dataset = namedtuple("Dataset", ["train", "val", "test","top"])

def load_image(mat_path):
    d = loadmat(mat_path)
    return d["image"], d["gender"][0], d["age"][0], d["db"][0], d["img_size"][0, 0], d["min_score"][0, 0]

sigmoid = non_matplotlib_utils.sigmoid

def load_fraction_best(average: bool = True):
    data_dir = os.path.realpath(os.path.join(__file__, "../../figures/plot_data"))
    plot_data = []
    for fname in [f"{data_dir}/{fname}" for fname in os.listdir(data_dir)]:
        with open(fname, "rb") as f:
            plot_data.append(pickle.load(f))

    fraction_best = {}
    for plot_datum in plot_data:
        for key, (mean, std) in plot_datum.items():
            assert (std == 0).all()
            if key in fraction_best:
                fraction_best[key].append(mean)
            else:
                fraction_best[key] = [mean]

    if average:
        for key, mean in fraction_best.items():
            fraction_best[key] = (
                np.mean(fraction_best[key], axis=0),
                np.std(fraction_best[key], axis=0),
            )
    return fraction_best


def plot_performance(
    fraction_best: Dict[str, Sequence[np.ndarray]],
    top_k_percent: int = 1,
    averaged: bool = True,
) -> None:
    if averaged:
        for model_name in fraction_best:
            mean, std = fraction_best[model_name]
            plt.plot(mean, label=model_name)
            plt.fill_between(range(len(mean)), mean - std, mean + std, alpha=0.5)
    else:
        for i, model_name in enumerate(fraction_best):
            for j, line in enumerate(fraction_best[model_name]):
                label = model_name if j == 0 else None
                plt.plot(line, label=label, color=f"C{i}", alpha=0.7)

    plt.xlabel("# Batches")
    plt.ylabel(f"Fraction of top {top_k_percent}% discovered")
    plt.title("Malaria Dataset")
    plt.legend()


def plot_contours(
    funcs: Union[Sequence[Union[Callable, np.ndarray]], Union[Callable, np.ndarray]],
    bounds,
    cmap: str = "viridis",
    n_points: int = 100,
    titles: Optional[Sequence[str]] = None,
):
    funcs = funcs if isinstance(funcs, Sequence) else [funcs]
    xs = np.linspace(*bounds[0], n_points)
    ys = np.linspace(*bounds[1], n_points)
    xs, ys = np.meshgrid(xs, ys)
    grid = np.stack((xs, ys), axis=-1)

    fig = plt.figure(figsize=(8 * len(funcs), 6))

    for i, func in enumerate(funcs):
        if isinstance(func, Callable):
            zs = func(grid)
        else:
            zs = func

        ax = fig.add_subplot(1, len(funcs), i + 1, projection="3d")
        ax.plot_surface(xs, ys, zs, cmap=cmap, rstride=1, cstride=1)
        ax.contour(xs, ys, zs, 10, offset=zs.min(), cmap=cmap)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        if titles:
            ax.set_title(titles[i])
    return ax


def random_points(bounds: np.ndarray, ndim: int, n_points: int) -> np.ndarray:
    """
    :param bounds: [lower_bounds, upper_bounds] where the bounds
      are arrays of the lower/upper bound for each dimension
    """

    lower_bounds = bounds[:, 0][np.newaxis, :]
    upper_bounds = bounds[:, 1][np.newaxis, :]

    points = lower_bounds + np.random.random(size=(n_points, ndim)) * (
        upper_bounds - lower_bounds
    )
    return points


<<<<<<< HEAD
def get_path(*path_segments):
    return os.path.realpath(os.path.join(*[str(segment) for segment in path_segments]))


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


def save_checkpoint(fname: str, model, optimizer: Optional = None) -> None:
    os.makedirs(os.path.dirname(fname), exist_ok=True)

    checkpoint = {"model_state": model.state_dict()}
    if optimizer:
        checkpoint["optimizer_state"] = optimizer.state_dict()

    torch.save(checkpoint, fname)


def load_checkpoint(fname: str, model, optimizer: Optional = None) -> None:
    checkpoint = torch.load(fname,map_location='cpu')
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
=======
get_path = non_matplotlib_utils.get_path
logp = non_matplotlib_utils.logp
qed = non_matplotlib_utils.qed
sas = non_matplotlib_utils.sas
all_scores = non_matplotlib_utils.all_scores
>>>>>>> fd1c8f0a8b88c4ece83c1aa9eaca6f078ee9ef25

train_val_test_split = non_matplotlib_utils.train_val_test_split
save_checkpoint = non_matplotlib_utils.save_checkpoint
load_checkpoint = non_matplotlib_utils.load_checkpoint
collated_expand = non_matplotlib_utils.collated_expand
make_batches = non_matplotlib_utils.make_batches
randint = non_matplotlib_utils.randint
make_contiguous_idx = non_matplotlib_utils.make_contiguous_idx
cumsub = non_matplotlib_utils.cumsub
sqdist = non_matplotlib_utils.sqdist

def jointplot(
    x,
    y,
    axis_labels: Tuple[str, str] = ("Predicted", "True"),
    title: str = "",
    **kwargs,
):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()

    if isinstance(y, torch.Tensor):
        y = y.detach().cpu().numpy()

    ax = sns.jointplot(x, y, s=3, alpha=0.5,stat_func=pearsonr, **kwargs)
    ax.set_axis_labels(*axis_labels)
    ax.ax_marg_x.set_title(title)
    return ax

def load_data(
    data_root: str,
    project: str,
    dataset: str,
    train_size,
    val_size: None,
    standardize_labels: bool = False,
    random_state: int = 521,
    device: Optional = None,
) -> _Dataset:
    device = device or "cpu"

    data_dir = get_path(data_root, project, dataset)
    inputs = np.load(get_path(data_dir, "inputs.npy"))
    labels = np.load(get_path(data_dir, "labels.npy"))

    train_inputs, test_inputs, train_labels, test_labels = train_test_split(
        inputs, labels, train_size=train_size, random_state=random_state
    )

    if val_size:
        train_inputs, val_inputs, train_labels, val_labels = train_test_split(
            train_inputs, train_labels, test_size=val_size, random_state=random_state
        )
    else:
        val_inputs = val_labels = None

    if standardize_labels:
        train_label_mean = train_labels.mean()
        train_label_std = train_labels.std()

        train_labels = (train_labels - train_label_mean) / train_label_std
        test_labels = (test_labels - train_label_mean) / train_label_std

        if val_inputs is not None:
            val_labels = (val_labels - train_label_mean) / train_label_std

    train_inputs = torch.tensor(train_inputs).float().to(device)
    train_labels = torch.tensor(train_labels).float().to(device)

    if val_inputs is not None:
        val_inputs = torch.tensor(val_inputs).float().to(device)
        val_labels = torch.tensor(val_labels).float().to(device)

    test_inputs = torch.tensor(test_inputs).float().to(device)
    test_labels = torch.tensor(test_labels).float().to(device)

    dataset = _Dataset(
        *[
            _Input_Labels(inputs, labels)
            for inputs, labels in zip(
                [train_inputs, val_inputs, test_inputs],
                [train_labels, val_labels, test_labels],
            )
        ]
    )
    return dataset

def load_data_saber(
    data_root: str,
    project: str,
    dataset: str,
    train_size,
    val_size: None,
    standardize_labels: bool = False,
    random_state: int = 521,
    device: Optional = None,
) -> _Dataset:
    device = device or "cpu"

    data_dir = get_path(data_root, project, dataset)
    inputs = np.load(get_path(data_dir, "inputs.npy"))
    labels = np.load(get_path(data_dir, "labels.npy"))
    labels = np.log(labels)

    train_inputs, test_inputs, train_labels, test_labels = train_test_split(
        inputs, labels, train_size=train_size, random_state=random_state
    )

    if val_size:
        train_inputs, val_inputs, train_labels, val_labels = train_test_split(
            train_inputs, train_labels, test_size=val_size, random_state=random_state
        )
    else:
        val_inputs = val_labels = None

    if standardize_labels:

        train_label_mean = train_labels.mean()
        train_label_std = train_labels.std()

        train_labels = sigmoid((train_labels - train_label_mean) / train_label_std)
        test_labels = sigmoid((test_labels - train_label_mean) / train_label_std)

        if val_inputs is not None:
            val_labels =sigmoid((val_labels - train_label_mean) / train_label_std)

    train_inputs = torch.tensor(train_inputs).float().to(device)
    train_labels = torch.tensor(train_labels).float().to(device)

    if val_inputs is not None:
        val_inputs = torch.tensor(val_inputs).float().to(device)
        val_labels = torch.tensor(val_labels).float().to(device)

    test_inputs = torch.tensor(test_inputs).float().to(device)
    test_labels = torch.tensor(test_labels).float().to(device)

    top_inputs=top_labels=None

    dataset = _Dataset(
        *[
            _Input_Labels(inputs, labels)
            for inputs, labels in zip(
                [train_inputs, val_inputs, test_inputs, top_inputs],
                [train_labels, val_labels, test_labels, top_labels],
            )
        ]
    )
    return dataset

def load_data_saber2(
    data_root: str,
    project: str,
    dataset: str,
    train_size,
    val_size: None,
    top_percent:float=0.1,
    log_transform:bool=True,
    standardize_labels: bool = True,
    random_state: int = 153,#521,
    device: Optional = None,
) -> _Dataset:
    device = device or "cpu"

    data_dir = get_path(data_root, project, dataset)
    inputs = np.load(get_path(data_dir, "inputs.npy"))
    labels = np.load(get_path(data_dir, "labels.npy"))
    if log_transform:
        labels = np.log(labels)

    indices = np.arange(labels.shape[0])

    labels_sort_idx = labels.argsort()
    sort_idx = labels_sort_idx[:-int(labels.shape[0]*top_percent)]
    #sort_idx = labels_sort_idx[:-int(labels.shape[0]*0.3)]
    indices = indices[sort_idx]
    inputs_l = inputs[indices]
    labels_l = labels[indices]
    top_inputs = inputs[labels_sort_idx[-int(labels.shape[0]*top_percent):]]
    top_labels = labels[labels_sort_idx[-int(labels.shape[0]*top_percent):]]

    train_inputs, test_inputs, train_labels, test_labels = train_test_split(
        inputs_l, labels_l, train_size=train_size, random_state=random_state
    )

    if val_size:
        train_inputs, val_inputs, train_labels, val_labels = train_test_split(
            train_inputs, train_labels, test_size=val_size, random_state=random_state
        )
    else:
        val_inputs = val_labels = None

    if standardize_labels:

        train_label_mean = train_labels.mean()
        train_label_std = train_labels.std()

        train_labels = sigmoid((train_labels - train_label_mean) / train_label_std)
        test_labels = sigmoid((test_labels - train_label_mean) / train_label_std)
        top_labels = sigmoid((top_labels - train_label_mean) / train_label_std)

        if val_inputs is not None:
            val_labels =sigmoid((val_labels - train_label_mean) / train_label_std)

    train_inputs = torch.tensor(train_inputs).float().to(device)
    train_labels = torch.tensor(train_labels).float().to(device)

    if val_inputs is not None:
        val_inputs = torch.tensor(val_inputs).float().to(device)
        val_labels = torch.tensor(val_labels).float().to(device)

    test_inputs = torch.tensor(test_inputs).float().to(device)
    test_labels = torch.tensor(test_labels).float().to(device)

    top_inputs = torch.tensor(top_inputs).float().to(device)
    top_labels = torch.tensor(top_labels).float().to(device)

    dataset = _Dataset(
        *[
            _Input_Labels(inputs, labels)
            for inputs, labels in zip(
                [train_inputs, val_inputs, test_inputs, top_inputs],
                [train_labels, val_labels, test_labels, top_labels],
            )
        ]
    )
    return dataset

def load_data_top(
    data_root: str,
    project: str,
    dataset: str,
    train_size,
    val_size: None,
    standardize_labels: bool = False,
    random_state: int = 521,
    device: Optional = None,
) -> _Dataset:
    device = device or "cpu"

    data_dir = get_path(data_root, project, dataset)
    inputs = np.load(get_path(data_dir, "inputs.npy"))
    labels = np.load(get_path(data_dir, "labels.npy"))

    train_inputs, test_inputs, train_labels, test_labels = top_split(
        inputs, labels, train_size=train_size, random_state=random_state
    )

    if val_size:
        train_inputs, val_inputs, train_labels, val_labels = train_test_split(
            train_inputs, train_labels, test_size=val_size, random_state=random_state
        )
    else:
        val_inputs = val_labels = None

    if standardize_labels:
        train_label_mean = train_labels.mean()
        train_label_std = train_labels.std()

        train_labels = (train_labels - train_label_mean) / train_label_std
        test_labels = (test_labels - train_label_mean) / train_label_std

        if val_inputs is not None:
            val_labels = (val_labels - train_label_mean) / train_label_std

    train_inputs = torch.tensor(train_inputs).float().to(device)
    train_labels = torch.tensor(train_labels).float().to(device)

    if val_inputs is not None:
        val_inputs = torch.tensor(val_inputs).float().to(device)
        val_labels = torch.tensor(val_labels).float().to(device)

    test_inputs = torch.tensor(test_inputs).float().to(device)
    test_labels = torch.tensor(test_labels).float().to(device)

    dataset = _Dataset(
        *[
            _Input_Labels(inputs, labels)
            for inputs, labels in zip(
                [train_inputs, val_inputs, test_inputs, top_inputs],
                [train_labels, val_labels, test_labels, top_inputs],
            )
        ]
    )
    return dataset

def load_data_wiki(
    data_root: str,
    project: str,
    dataset: str,
    train_size,
    val_size: None,
    top_percent:float=None,
    standardize_labels: bool = True,
    random_state: int = 521,
    device: Optional = None,
) -> _Dataset:
    device = device or "cpu"

    data_dir = join(data_root,dataset+'_db.mat')
    image,gender,labels, _, _, _ = load_image(data_dir)
    inputs=np.moveaxis(image, -1, 1)
    indices = np.arange(labels.shape[0])

    if top_percent: 
        labels_sort_idx = labels.argsort()
        sort_idx = labels_sort_idx[:-int(labels.shape[0]*top_percent)]
        indices = indices[sort_idx]
        inputs_l = inputs[indices]
        labels_l = labels[indices]
        top_inputs = inputs[labels_sort_idx[-int(labels.shape[0]*top_percent):]]
        top_labels = labels[labels_sort_idx[-int(labels.shape[0]*top_percent):]]
    else:
        male_idx=np.where(gender==1)
        female_idx=np.where(gender==0)
        inputs_l = inputs[male_idx]
        labels_l = labels[male_idx]
        top_inputs = inputs[female_idx]
        top_labels = labels[female_idx]

    train_inputs, test_inputs, train_labels, test_labels = train_test_split(
        inputs_l, labels_l, train_size=train_size, random_state=random_state
    )

    if val_size:
        train_inputs, val_inputs, train_labels, val_labels = train_test_split(
            train_inputs, train_labels, test_size=val_size, random_state=random_state
        )
    else:
        val_inputs = val_labels = None

    if standardize_labels:

        train_label_mean = train_labels.mean()
        train_label_std = train_labels.std()

        train_labels = sigmoid((train_labels - train_label_mean) / train_label_std)
        test_labels = sigmoid((test_labels - train_label_mean) / train_label_std)
        top_labels = sigmoid((top_labels - train_label_mean) / train_label_std)

        if val_inputs is not None:
            val_labels =sigmoid((val_labels - train_label_mean) / train_label_std)

    train_inputs = torch.tensor(train_inputs).float().to(device)
    train_labels = torch.tensor(train_labels).float().to(device)

    if val_inputs is not None:
        val_inputs = torch.tensor(val_inputs).float().to(device)
        val_labels = torch.tensor(val_labels).float().to(device)

    test_inputs = torch.tensor(test_inputs).float().to(device)
    test_labels = torch.tensor(test_labels).float().to(device)

    top_inputs = torch.tensor(top_inputs).float().to(device)
    top_labels = torch.tensor(top_labels).float().to(device)

    dataset = _Dataset(
        *[  
            _Input_Labels(inputs, labels)
            for inputs, labels in zip(
                [train_inputs, val_inputs, test_inputs, top_inputs],
                [train_labels, val_labels, test_labels, top_labels],
            )
        ]
    )
    return dataset

get_early_stopping = non_matplotlib_utils.get_early_stopping

def load_data_maxvar(data_dir, exclude_top,standardize_labels=True, device=None):
    device = device or "cpu"
    data_dir = data_dir + "/"

    train_inputs = np.load(data_dir+'inputs_crx_ref_r1_train_' + str(exclude_top) + '.npy')
    val_inputs = np.load(data_dir+'inputs_crx_ref_r1_val_' + str(exclude_top) + '.npy')
    test_inputs = np.load(data_dir+'inputs_crx_ref_r1_test_' + str(exclude_top) + '.npy')

    train_labels = np.load(data_dir+'labels_crx_ref_r1_train_' + str(exclude_top) + '.npy')
    val_labels = np.load(data_dir+'labels_crx_ref_r1_val_' + str(exclude_top) + '.npy')
    test_labels = np.load(data_dir+'labels_crx_ref_r1_test_' + str(exclude_top) + '.npy')

    if standardize_labels:

        train_label_mean = train_labels.mean()
        train_label_std = train_labels.std()

        train_labels = sigmoid((train_labels - train_label_mean) / train_label_std)
        test_labels = sigmoid((test_labels - train_label_mean) / train_label_std)

        if val_inputs is not None:
            val_labels =sigmoid((val_labels - train_label_mean) / train_label_std)

    train_inputs = torch.tensor(train_inputs).float().to(device)
    val_inputs = torch.tensor(val_inputs).float().to(device)
    test_inputs = torch.tensor(test_inputs).float().to(device)

    train_labels = torch.tensor(train_labels).float().to(device)
    val_labels = torch.tensor(val_labels).float().to(device)
    test_labels = torch.tensor(test_labels).float().to(device)

    dataset = _Dataset(
        *[
            _Input_Labels(inputs, labels)
            for inputs, labels in zip(
                [train_inputs, val_inputs, test_inputs],
                [train_labels, val_labels, test_labels],
            )
        ]
    )
    return dataset
