
"""ops.py"""

import torch
import torch.nn.functional as F


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
    return x.sum().item() > 1e-5

def is_inf(x):
    return tensor_any(x == float('inf')) and tensor_any(x == -float('inf'))

def is_nan(x):
    return tensor_any(x != x)

def is_finite(x):
    return tensor_all((x != float('inf')) * (x == x))
