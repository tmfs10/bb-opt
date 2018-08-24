
import torch
import numpy as np
import math

class Shekel(object):

    """ values taken from https://www.sfu.ca/~ssurjano/shekel.html """
    def __init__(self, args):
        self.beta = 0.1*torch.Tensor([1, 2, 2, 4, 4, 6, 3, 7, 5, 5]).to(args.device)
        self.C = torch.Tensor(
                [4, 4, 4, 4],
                [1, 1, 1, 1],
                [8, 8, 8, 8],
                [6, 6, 6, 6],
                [3, 7, 3, 7],
                [2, 9, 2, 9],
                [5, 3, 5, 3],
                [8, 1, 8, 1],
                [6, 2, 6, 2],
                [7, 3.6, 7, 3.6]).to(args.device)

    def __call__(self, x, m=10):
        assert m >= 1
        assert m <= 10
        return -torch.sum((torch.sum((x-self.C[:m])**2, dim=1) + self.beta[:m])**-1)


class Michalewicz(object):

    def __call__(self, x, m=10):
        d = x.shape[0]
        if m % 2 == 0:
            return -torch.sum(torch.sin(x) * (torch.sinh(x**2/math.pi)**(2*m)))
        else:
            return torch.sum(torch.sin(x) * (torch.sinh(x**2/math.pi)**(2*m)))


class Eggholder(object):

    def __call__(self, x):
        assert x.shape[0] == 2
        return -(x[1]+47)*torch.sin(torch.sqrt(torch.abs(x[1]+x[0]/2+47)))-x[0]*torch.sin(torch.sqrt(torch.abs(x[0]-(x[1]+47))))
