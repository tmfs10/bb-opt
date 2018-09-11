import torch
import sys
from typing import Sequence


def dimwise_mixrbf_kernels(X, bws=[.01, .1, .2, 1, 5, 10, 100], wts=None):
    """Mixture of RBF kernels between each dimension of X.

    If X is shape (n, d), returns shape (n, n, d).

    Kernel is sum_i wt_i exp(- (x - y)^2 / (2 bw^2)).
    If wts is not passed, uses 1 for each alpha.
    """
    shp = X.shape
    assert len(shp) == 2

    bws = torch.tensor(bws, dtype=X.dtype)
    wts = ((1./len(bws) if wts is None else wts) * torch.ones(bws.shape)).type(X.type())

    sqdists = (torch.unsqueeze(X, 0) - torch.unsqueeze(X, 1)) ** 2
    bws_e = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(bws, 1), 2), 3)

    parts = torch.exp(torch.unsqueeze(sqdists, 0) / (-2 * bws_e ** 2))
    return torch.einsum("w,wijd->ijd", (wts, parts))


def dimwise_mixrq_kernels(
    X: torch.Tensor, alphas: Sequence[float] = (.2, .5, 1, 2, 5), weights=None
) -> torch.Tensor:
    """
    Mixture of RQ kernels between each dimension of X.

    If X is shape (n, d), returns shape (n, n, d).
    n is the number of samples from each RV and d is the number of RVs.
    k_ijk = RQK(X[i, k], X[j, k]) (the kernel of the i'th and j'th samples of RV k).

    Kernel is sum_i wt_i (1 + (x - y)^2 / (2 alpha_i))^{-alpha_i}.
    If weights is not passed, each alpha is weighted equally.

    A vanilla RQ kernel is k(x, x') = sigma^2 (1 + (x - x')^2 / (2 alpha l^2)) ^ -alpha
    Here we (by defautl) use a weighted combination of multiple alphas and set l = sigma = 1.
    """

    assert X.ndimension() == 2, X.ndimension()

    alphas = X.new_tensor(alphas).view(-1, 1, 1, 1)
    weights = weights or 1.0 / len(alphas)
    weights = weights * X.new_ones(len(alphas))

    # dims are (alpha, x, y, dim)
    sqdists = ((X.unsqueeze(0) - X.unsqueeze(1)) ** 2).unsqueeze(0)
    # 30% faster without asserts in some quick tests (n = 250, d = 1K)
    #     assert (sqdists >= 0).all()

    logs = torch.log1p(sqdists / (2 * alphas))
    #     assert torch.isfinite(logs).all()
    return torch.einsum("w,wijd->ijd", (weights, torch.exp(-alphas * logs)))


def linear_kernel(x: torch.Tensor, c: float = 1):
    """
    :param x: n x d for n samples from d RVs
    """
    return x.unsqueeze(0) * x.unsqueeze(1) + c


################################################################################
# HSIC estimators


def total_hsic(kernels, logspace=True):
    """(Biased) estimator of total independence.

    kernels should be shape (n, n, d) to test for total
    independence among d variables with n paired samples.
    """
    # formula from section 4.4 of
    # https://papers.nips.cc/paper/4893-a-kernel-test-for-three-variable-interactions.pdf
    shp = kernels.shape
    assert len(shp) == 3
    assert shp[0] == shp[1], "%s == %s" % (str(shp[0]), str(shp[1]))

    n = kernels.new_tensor(shp[0])
    d = kernels.new_tensor(shp[2])

    # t1: 1/n^2      sum_a  sum_b  prod_i K_ab^i
    # t2: -2/n^(d+1) sum_a  prod_i sum_b  K_ab^i
    # t3: 1/n^(2d)   prod_i sum_a  sum_b  K_ab^i

    if not logspace:
        sum_b = torch.sum(kernels, dim=1)

        t1 = torch.mean(torch.prod(kernels, dim=2))
        t2 = torch.sum(torch.prod(sum_b, dim=1)) * (-2 / (n ** (d + 1)))
        t3 = torch.prod(torch.sum(sum_b, dim=0)) / (n ** (2 * d))
        return t1 + t2 + t3
    else:
        log_n = torch.log(n)
        log_2 = torch.log(kernels.new_tensor(2))
        log_kernels = kernels.log_()  # TODO: just take directly?
        log_sum_b = log_kernels.logsumexp(dim=1)

        l1 = log_kernels.sum(dim=2).logsumexp(dim=1).logsumexp(dim=0) - 2 * log_n
        l2 = log_sum_b.sum(dim=1).logsumexp(dim=0) + log_2 - (d + 1) * log_n
        l3 = log_sum_b.logsumexp(dim=0).sum() - 2 * d * log_n

        # total_hsic = exp(l1) - exp(l2) + exp(l3)
        #   = exp(-a) (exp(l1 + a) - exp(l2 + a) + exp(l3 + a)) for any a
        # can't use logsumexp for this directly because we subtract the l2 term
        a = torch.max(kernels.new_tensor([l1, l2, l3]))
        return a.exp() * ((l1 - a).exp() - (l2 - a).exp() + (l3 - a).exp())


def sum_pairwise_hsic(kernels):
    """Sum of (biased) estimators of pairwise independence.

    kernels should be shape (n, n, d) when testing
    among d variables with n paired samples.
    """
    shp = kernels.shape
    assert len(shp) == 3
    assert shp[0] == shp[1], "%s == %s" % (str(shp[0]), str(shp[1]))

    n = torch.tensor(shp[0], dtype=kernels.dtype)

    # Centered kernel matrix is given by:
    # (I - 1/n 1 1^T) K (I - 1/n 1 1^T)
    #  = K - 1/n 1 1^T K - 1/n K 1 1^T + 1/n^2 1 1^T K 1 1^T
    row_means = torch.mean(kernels, dim=0, keepdims=True)
    col_means = torch.mean(kernels, dim=1, keepdims=True)
    grand_mean = torch.mean(row_means, dim=1, keepdims=True)
    centered_kernels = kernels - row_means - col_means + grand_mean

    # HSIC on dims (i, j) is  1/n^2 sum_{a, b} K[a, b, i] K[a, b, j]
    # sum over all dimensions is 1/n^2 sum_{i, j, a, b} K[a, b, i] K[a, b, j]
    return torch.einsum("abi,abj->", (centered_kernels, centered_kernels)) / n ** 2


def precompute_batch_hsic_stats(
    preds: torch.Tensor, batch: Sequence[int]
) -> Sequence[torch.Tensor]:
    batch_kernels = linear_kernel(preds[:, batch])
    batch_kernels.log_()
    batch_kernels = batch_kernels.unsqueeze(-1)

    batch_sum_b = torch.logsumexp(batch_kernels, dim=1)
    batch_l1_sum = torch.sum(batch_kernels, dim=2)
    batch_l2_sum = batch_sum_b.sum(dim=1)
    batch_l3_sum = batch_sum_b.logsumexp(dim=0).sum(dim=0)
    return batch_sum_b, batch_l1_sum, batch_l2_sum, batch_l3_sum


def compute_point_hsics(
    preds: torch.Tensor,
    next_points: Sequence[int],
    batch_sum_b: torch.Tensor,
    batch_l1_sum: torch.Tensor,
    batch_l2_sum: torch.Tensor,
    batch_l3_sum: torch.Tensor,
) -> torch.Tensor:
    log_n = preds.new_tensor(len(batch_sum_b)).log_()
    d = preds.new_tensor(batch_sum_b.shape[1] + 1)
    log_2 = preds.new_tensor(2).log_()

    point_kernels = linear_kernel(preds[:, next_points])
    point_kernels.log_()
    point_kernels = point_kernels.unsqueeze(2)

    point_sum_b = point_kernels.logsumexp(dim=1)
    point_l1_sum = point_kernels.sum(dim=2)
    point_l2_sum = point_sum_b.sum(dim=1)
    point_l3_sum = point_sum_b.logsumexp(dim=0).sum(dim=0)

    l1 = (batch_l1_sum + point_l1_sum).logsumexp(dim=1).logsumexp(dim=0) - 2 * log_n
    l2 = (batch_l2_sum + point_l2_sum).logsumexp(dim=0) + log_2 - (d + 1) * log_n
    l3 = batch_l3_sum + point_l3_sum - 2 * d * log_n

    a = -torch.stack((l1, l2, l3)).max(dim=0)[0]

    hsics = torch.exp(-a) * (torch.exp(l1 + a) - torch.exp(l2 + a) + torch.exp(l3 + a))
    return hsics
