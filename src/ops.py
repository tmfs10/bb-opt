
"""ops.py"""

import torch
import torch.nn.functional as F

_eps = 1.0e-5


def recon_loss(x, x_recon):
    n = x.size(0)
    loss = F.binary_cross_entropy_with_logits(x_recon, x, size_average=False).div(n)
    return loss


def kl_divergence(mu, logvar):
    kld = -0.5*(1+logvar-mu**2-logvar.exp()).sum(1).mean()
    return kld


def permute_dims(z):
    assert z.dim() == 2

    B, _ = z.size()
    perm_z = []
    for z_j in z.split(1, 1):
        perm = torch.randperm(B).to(z.device)
        perm_z_j = z_j[perm]
        perm_z.append(perm_z_j)

    return torch.cat(perm_z, 1)

def dot(x, y):
    return torch.squeeze(torch.matmul(tf.unsqueeze(x, 0), tf.unsqueeze(y, 1)))

def sq_sum(t):
    "The squared Frobenius-type norm of a tensor, sum(t ** 2)."
    return 2 * torch.sum(t**2)

def tensor_all(x):
    return (1-(x.long())).sum().item() < 1e-5

def tensor_any(x):
    return x.long().sum().item() > 1e-5

def is_inf(x):
    return tensor_any(x == float('inf')) and tensor_any(x == -float('inf'))

def isinf(x):
    return is_inf(x)

def is_nan(x):
    return tensor_any(x != x)

def isnan(x):
    return is_nan(x)

def is_finite(x):
    return tensor_all((x != float('inf')) * (x == x))

def set_diagonal(x, v):
    assert len(x.shape) == 2
    assert len(v.shape) == 1
    assert x.shape[0] == v.shape[0]
    assert x.shape[1] == v.shape[0]
    mask = torch.diag(torch.ones_like(v))
    x = mask*torch.diag(v) + (1-mask)*x
    return x

def pearsonr(x, y):
    """
    Mimics `scipy.stats.pearsonr`

    Arguments
    ---------
    x : 1D torch.Tensor
    y : 1D torch.Tensor

    Returns
    -------
    r_val : float
        pearsonr correlation coefficient between x and y
    
    Scipy docs ref:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html
    
    Scipy code ref:
        https://github.com/scipy/scipy/blob/v0.19.0/scipy/stats/stats.py#L2975-L3033
    Example:
        >>> x = np.random.randn(100)
        >>> y = np.random.randn(100)
        >>> sp_corr = scipy.stats.pearsonr(x, y)[0]
        >>> th_corr = pearsonr(torch.from_numpy(x), torch.from_numpy(y))
        >>> np.allclose(sp_corr, th_corr)
    """
    mean_x = torch.mean(x)
    mean_y = torch.mean(y)
    xm = x.sub(mean_x)
    ym = y.sub(mean_y)
    r_num = xm.dot(ym)
    r_den = torch.norm(xm, 2) * torch.norm(ym, 2)
    r_val = r_num / r_den
    return r_val

def corrcoef(x):
    """
    Mimics `np.corrcoef`

    Arguments
    ---------
    x : 2D torch.Tensor
    
    Returns
    -------
    c : torch.Tensor
        if x.size() = (5, 100), then return val will be of size (5,5)

    Numpy docs ref:
        https://docs.scipy.org/doc/numpy/reference/generated/numpy.corrcoef.html
    Numpy code ref: 
        https://github.com/numpy/numpy/blob/v1.12.0/numpy/lib/function_base.py#L2933-L3013

    Example:
        >>> x = np.random.randn(5,120)
        # result is a (5,5) matrix of correlations between rows
        >>> np_corr = np.corrcoef(x)
        >>> th_corr = corrcoef(torch.from_numpy(x))
        >>> np.allclose(np_corr, th_corr.numpy())
        # [out]: True
    """
    # calculate covariance matrix of rows
    mean_x = x.mean(1, keepdim=True)
    xm = x.sub(mean_x.expand_as(x))
    c = xm.mm(xm.t())
    c = c / (x.size(1) - 1)

    # normalize covariance matrix
    d = torch.diag(c)
    stddev = torch.pow(d, 0.5)
    c = c.div(stddev.expand_as(c))
    c = c.div(stddev.expand_as(c).t())

    # clamp between -1 and 1
    # probably not necessary but numpy does it
    c = torch.clamp(c, -1.0, 1.0)

    return c

def cross_corrcoef(x, y):
    # x is (n, p), y is (n, q)
    # calculate covariance matrix of rows
    mean_x = x.mean(0, keepdim=True)
    mean_y = y.mean(0, keepdim=True)
    xm = x.sub(mean_x.expand_as(x))
    ym = y.sub(mean_y.expand_as(y))
    c = xm.t().mm(ym)
    c = c / (x.size(0) - 1)

    x_std = x.std(0, keepdim=True).transpose(0, 1)
    y_std = y.std(0, keepdim=True)
    norm = x_std*y_std

    c  = c/norm

    # clamp between -1 and 1
    # probably not necessary but numpy does it
    c = torch.clamp(c, -1.0, 1.0)

    return c
