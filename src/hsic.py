import torch
import global_constants as gl
import sys
import ops

def dimwise_mixrbf_kernels(X, bws=[.01, .1, .2, 1, 5, 10, 100], wts=None):
    """Mixture of RBF kernels between each dimension of X.

    If X is shape (n, d), returns shape (n, n, d).

    Kernel is sum_i wt_i exp(- (x - y)^2 / (2 bw^2)).
    If wts is not passed, uses 1 for each alpha.
    """
    shp = X.shape
    assert len(shp) == 2

    bws = torch.tensor(bws, dtype=X.dtype)
    wts = ((1 if wts is None else wts) * torch.ones(bws.shape)).type(X.type())

    sqdists = torch.square(torch.unsqueeze(X, 0) - torch.unsqueeze(X, 1))
    bws_e = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(bws, 1), 2), 3)
    wts_e = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(wts, 1), 2), 3)

    parts = torch.exp(torch.unsqueeze(sqdists, 0) / (-2 * bws_e ** 2))
    return torch.sum(wts_e * parts, 0)


def dimwise_mixrq_kernels(X, alphas=[.2, .5, 1, 2, 5], wts=None):
    """Mixture of RQ kernels between each dimension of X.

    If X is shape (n, d), returns shape (n, n, d).

    Kernel is sum_i wt_i (1 + (x - y)^2 / (2 alpha_i))^{-alpha_i}.
    If wts is not passed, uses 1 for each alpha.
    """
    shp = X.shape
    assert len(shp) == 2, len(shp)

    alphas = torch.tensor(alphas).type(X.type())
    wts = ((1 if wts is None else wts)*torch.ones(alphas.shape)).type(X.type())

    # dims are (alpha, x, y, dim)
    sqdists = torch.unsqueeze((torch.unsqueeze(X, 0) - torch.unsqueeze(X, 1))**2, 0)
    assert ops.tensor_all(sqdists >= 0)
    alphas_e = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(alphas, 1), 2), 3)

    logs = torch.log1p(sqdists / (2 * alphas_e))
    assert ops.is_finite(logs)
    return torch.einsum('w,wijd->ijd', (wts, torch.exp(-alphas_e * logs)))


################################################################################
# HSIC estimators

def total_hsic(kernels):
    """(Biased) estimator of total independence.

    kernels should be shape (n, n, d) to test for total
    independence among d variables with n paired samples.
    """
    # formula from section 4.4 of
    # https://papers.nips.cc/paper/4893-a-kernel-test-for-three-variable-interactions.pdf
    kernels = kernels.type(gl.DoubleTensor)
    shp = kernels.shape
    assert len(shp) == 3
    assert shp[0] == shp[1], "%s == %s" % (str(shp[0]), str(shp[1]))

    n = torch.tensor(shp[0], dtype=kernels.dtype).type(kernels.type())
    d = torch.tensor(shp[2], dtype=kernels.dtype).type(kernels.type())

    sum_b = torch.sum(kernels, dim=1)
    t1 = torch.mean(torch.prod(kernels, dim=2))
    t2 = - 2 / (n ** (d + 1)) * torch.sum(torch.prod(sum_b, dim=1))

    # do this third term in logspace:
    sum_ab = torch.sum(sum_b, dim=0)
    t3 = torch.exp(-2 * d * torch.log(n) + torch.sum(torch.log(sum_ab)))

    #print >> sys.stderr, "t1:", t1.item(), "; t2:", t2.item(), "; t3", t3.item()
    #print >> sys.stderr, "kernels:", kernels[:3, :3, :3]

    return (t1 + t2 + t3).type(gl.FloatTensor)


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
    return torch.einsum('abi,abj->', (centered_kernels, centered_kernels)) / n**2
