"""
Reimplementation in numpy of the MI estimators from https://github.com/BiuBiuBiLL/NPEET_LNC
"""

import numpy as np
from scipy.special import digamma
from sklearn.neighbors import KDTree


def compute_avg_digamma(points: np.ndarray, dvec: np.ndarray) -> float:
    # This part finds number of neighbors in some radius in the marginal space
    # returns expected value of <psi(nx)>
    tree = KDTree(points, p=float("inf"))

    # don't include the boundary point because we include the center point?
    neighbors = tree.query_radius(points, dvec - 1e-15)
    n_neighbors = np.array(list(map(len, neighbors)))
    avg_digamma = digamma(n_neighbors).mean()
    return avg_digamma


def estimate_mi(
    x: np.ndarray,
    k: int = 5,
    alpha: float = 0.25,
    noise_std: float = 1e-10,
    method: str = "LNC",
):
    """
    The mutual information estimator by Kraskov et al.
    :param x: of shape (n_vars, n_samples)
    :param method: one of {LNC, KSG}
    """
    assert method in ("LNC", "KSG"), f"Unrecognized method {method}"

    # First Step: calculate the mutual information using the Kraskov mutual information estimator
    # adding small noise to X, e.g., x<-X+noise
    n_vars, n_samples = x.shape

    # adding small noise to X, e.g., x<-X+noise
    x += noise_std * np.random.rand(n_vars, n_samples)
    x = x.T

    # find neighbors in joint distance
    tree = KDTree(x, p=float("inf"))
    neighbor_idx = tree.query(x, k + 1, return_distance=False)
    neighbors = x[neighbor_idx]

    # compute marginal distances for the furthest neighbor
    dvec = abs(neighbors[:, 1:] - neighbors[:, 0:1]).max(axis=1).T

    ret = 0

    for i in range(n_vars):
        ret -= compute_avg_digamma(x[:, i, None], dvec[i])

    ret += digamma(k) - (n_vars - 1) / k + (n_vars - 1) * digamma(n_samples)

    if method == "KSG":
        return ret

    # Second Step: Add the correction term (Local Non-Uniform Correction)

    # Substract mean of k-nearest neighbor points
    neighbors -= neighbors[:, 0:1]

    # Calculate covariance matrix of k-nearest neighbor points, obtain eigen vectors
    idx1 = [i for i in range(n_vars) for _ in range(n_vars)]  # e.g. [0, 0, 1, 1]
    idx2 = [i for _ in range(n_vars) for i in range(n_vars)]  # e.g. [0, 1, 0, 1]
    covr = (
        (neighbors[..., idx1] * neighbors[..., idx2])
        .mean(axis=1)
        .reshape((-1, n_vars, n_vars))
    )
    w, v = np.linalg.eigh(covr)

    # Calculate PCA-bounding box using eigen vectors
    V_rect = np.log(
        abs((v[:, None, ...] * neighbors[..., None]).sum(axis=2)).max(axis=1)
    ).sum(axis=1)

    # Calculate the volume of original box
    log_knn_dist = np.log(dvec).sum(axis=0)

    # Perform local non-uniformity checking and compute correction term
    correction_idx = (log_knn_dist + np.log(alpha) > V_rect) & (log_knn_dist > V_rect)
    correction = (
        log_knn_dist[correction_idx] - V_rect[correction_idx]
    ).sum() / n_samples

    return ret + correction
