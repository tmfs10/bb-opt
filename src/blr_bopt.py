
"""
Bayesian linear regression optimization

Based on https://arxiv.org/pdf/1502.05700.pdf

"""

import sys
import torch
import torch.nn as nn
import numpy as np

class AdaptiveBasisRegression(nn.Module):

    def __init__(self, num_dimensions):
        assert num_dimensions > 0
        self.num_dimensions = num_dimensions
        self.alpha = torch.zeros(num_dimensions)
        self.beta = torch.zeros(num_dimensions, num_dimensions)

    # phi - design matrix of shape NxD where N is # of points and D is # of dimensions
    def forward(self, phi, y):
        phi_T = torch.tranpose(phi, (0, 1))
        K = torch.mm(self.beta, torch.mm(phi_T, phi)) \
                + torch.mv(torch.eye(self.num_dimensions), self.alpha)
        m = torch.mv(
                torch.mm(self.beta, 
                    torch.mm(torch.inv(K), phi_T)),
                f
