import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import numpy as np
import os
import pickle
from typing import Dict, Tuple, Sequence, Union, Callable, Optional
from rdkit import Chem
from rdkit.Chem import Descriptors, QED
import sascorer


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


def get_path(*path_segments):
    return os.path.realpath(os.path.join(*path_segments))

def logp(smiles_mol, add_hs=False):
    m = Chem.MolFromSmiles(smiles_mol)
    return Descriptors.MolLogP(m, add_hs)

def qed(smiles_mol):
    m = Chem.MolFromSmiles(smiles_mol)
    return QED.qed(m)

def sas(smiles_mol):
    m = Chem.MolFromSmiles(smiles_mol)
    return sascorer.calculateScore(m)

def all_scores(smiles_mol, add_hs=False):
    m = Chem.MolFromSmiles(smiles_mol)
    return Descriptors.MolLogP(m, add_hs), QED.qed(m), sascorer.calculateScore(m)
