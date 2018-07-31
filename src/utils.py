import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import numpy as np
from typing import Dict, Tuple, Sequence, Union, Callable, Optional


def plot_performance(
    fraction_best: Dict[str, Tuple[np.ndarray, np.ndarray]], top_k_percent: int = 1
) -> None:
    for model_name in fraction_best:
        mean, std = fraction_best[model_name]
        plt.plot(mean, label=model_name)
        plt.fill_between(range(len(mean)), mean - std, mean + std, alpha=0.5)

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
