"""
Benchmark problems which correspond to so-called "classic" problems from the
global-optimization literature. All problems are transformed into maximization
problems and can be combined with additional additive noise.
Forked from https://github.com/mwhoffman/benchfunk/.
"""

import numpy as np
from typing import Optional

__all__ = [
    "Sinusoidal",
    "Gramacy",
    "Branin",
    "Bohachevsky",
    "Goldstein",
    "Hartmann3",
    "Hartmann6",
]


class Benchmark(object):
    """
    Base class for benchmark functions which should be MAXIMIZED. Each class
    which implements this interface should include a method self._f(x) which
    takes an (n,d)-array of input locations and returns the value of each
    input.
    """

    ndim = None

    def __init__(self, variance: float = 0.0, seed: Optional[float] = None):
        self._variance = variance
        self._rng = np.random.RandomState(seed)

    def __repr__(self):
        variance = f"(variance = {self._variance})" if self._variance > 0 else ""
        typename = type(self).__name__
        return f"{typename}{variance}"

    def _f(self, X):
        """Vectorized access to the noise-free benchmark function. Note: this
        function does no bounds checking."""
        raise NotImplementedError

    def __call__(self, x):
        """Vectorized access to the benchmark function."""
        if self.ndim is None:
            print("All subclasses of `Benchmark` must define `ndim`")
            raise NotImplementedError

        x = np.array(x, dtype=float, copy=False)  # convert in case of list
        if x.shape[-1] != self.ndim:
            raise ValueError(f"Last dimenion of inputs must be size {self.ndim}.")

        y = self._f(x)

        if self._variance > 0:
            y += self._rng.normal(scale=np.sqrt(self._variance), size=len(y))
        return y


class Sinusoidal(Benchmark):
    """
    Simple sinusoidal function bounded in [0, 2pi] given by -cos(x)-sin(3x).
    """

    bounds = np.array([[0, 2 * np.pi]])
    xopt = np.array([3.61439678])
    ndim = 1

    def _f(self, x):
        return -np.ravel(np.cos(x) + np.sin(3 * x))


class Gramacy(Benchmark):
    """
    Sinusoidal function in 1d used by Gramacy and Lee in "Cases for the nugget
    in modeling computer experiments".
    """

    bounds = np.array([[0.5, 2.5]])
    xopt = np.array([0.54856343])
    ndim = 1

    def _f(self, x):
        return -np.ravel(np.sin(10 * np.pi * x) / (2 * x) + (x - 1) ** 4)


class Branin(Benchmark):
    """
    The 2d Branin function bounded in [-5,10] to [0,15]. Global optimizers
    exist at [-pi, 12.275], [pi, 2.275], and [9.42478, 2.475] with no local
    optimizers.
    """

    bounds = np.array([[-5, 10.], [0, 15]])
    xopt = np.array([np.pi, 2.275])
    ndim = 2

    def _f(self, x):
        a = 1
        b = 5.1 / (4 * np.pi ** 2)
        c = 5 / np.pi
        r = 6
        s = 10
        t = 1 / (8 * np.pi)

        y = a * (x[..., 1] - b * x[..., 0] ** 2 + c * x[..., 0] - r) ** 2
        y += s * (1 - t) * np.cos(x[..., 0]) + s
        # negate to make this a maximization problem
        # NOTE: this rescales branin by 10 to make it more manageable.
        return -y / 10


class Bohachevsky(Benchmark):
    """
    The Bohachevsky function in 2d, bounded in [-100, 100] for both variables.
    There is only one global optimizer at [0, 0].
    """

    bounds = np.array([[-100, 100.], [-100, 100]])
    xopt = np.array([0, 0.])
    ndim = 2

    def _f(self, x):
        x0 = x[..., 0]
        x1 = x[..., 1]
        y = 0.7 + x0 ** 2 + 2.0 * x1 ** 2
        y -= 0.3 * np.cos(3 * np.pi * x0)
        y -= 0.4 * np.cos(4 * np.pi * x1)
        return -y


class Goldstein(Benchmark):
    """
    The Goldstein & Price function in 2d, bounded in [-2,-2] to [2,2]. There
    are several local optimizers and a single global optimizer at [0,-1].
    """

    bounds = np.array([[-2, 2.], [-2, 2]])
    xopt = np.array([0, -1.])
    ndim = 2

    def _f(self, x):
        a = 1 + (x[:, 0] + x[:, 1] + 1) ** 2 * (
            19
            - 14 * x[:, 0]
            + 3 * x[:, 0] ** 2
            - 14 * x[:, 1]
            + 6 * x[:, 0] * x[:, 1]
            + 3 * x[:, 1] ** 2
        )
        b = 30 + (2 * x[:, 0] - 3 * x[:, 1]) ** 2 * (
            18
            - 32 * x[:, 0]
            + 12 * x[:, 0] ** 2
            + 48 * x[:, 1]
            - 36 * x[:, 0] * x[:, 1]
            + 27 * x[:, 1] ** 2
        )
        return -a * b


class Hartmann3(Benchmark):
    """
    The 3-dimensional Hartmann function bounded in [0,1]^3.
    """

    ndim = 3
    bounds = np.array(3 * [[0., 1.]])
    xopt = np.array([0.114614, 0.555649, 0.852547])

    def _f(self, x):
        a = np.array(
            [[3.0, 10., 30.], [0.1, 10., 35.], [3.0, 10., 30.], [0.1, 10., 35.]]
        )[None]
        c = np.array([1., 1.2, 3., 3.2])[None]
        p = np.array(
            [
                [0.36890, 0.11700, 0.26730],
                [0.46990, 0.43870, 0.74700],
                [0.10910, 0.87320, 0.55470],
                [0.03815, 0.57430, 0.88280],
            ]
        )[None]
        return np.sum(c * np.exp(-np.sum(a * (x[:, None] - p) ** 2, -1)), -1)


class Hartmann6(Benchmark):
    """
    The 6-dimensional Hartmann function bounded in [0,1]^6.
    """

    ndim = 6
    bounds = np.array(6 * [[0., 1.]])
    xopt = np.array([0.201667, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573])

    def _f(self, x):
        a = np.array(
            [
                [10., 3.0, 17., 3.5, 1.7, 8.0],
                [.05, 10., 17., 0.1, 8.0, 14.],
                [3.0, 3.5, 1.7, 10., 17., 8.0],
                [17., 8.0, .05, 10., 0.1, 14.],
            ]
        )[None]
        c = np.array([1.0, 1.2, 3.0, 3.2])[None]
        p = np.array(
            [
                [0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886],
                [0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991],
                [0.2348, 0.1451, 0.3522, 0.2883, 0.3047, 0.6650],
                [0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381],
            ]
        )[None]
        F = np.sum(c * np.exp(-np.sum(a * (x[:, None] - p) ** 2, -1)), -1)
        # NOTE: this rescales the problem to have mean zero and variance 1, as
        # used in Picheny et al., 2012.
        F = (F + 2.58) / 1.94
        return F
