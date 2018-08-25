
import torch
import numpy as np
import math

"""
values taken from 
    * https://www.sfu.ca/~ssurjano/shekel.html 
    * http://benchmarkfcns.xyz/benchmarkfcns/ackleyfcn.html 
    * wikipedia
"""

class Shekel(object):
    """ 
    Global minima:
        m=5,7,10, at x*=(4,4,4,4)
    """

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
    """ 
    Global minima:
        d=2, x*=(2.2, 1.57)
    """

    def __call__(self, x, m=10):
        d = x.shape[0]
        if m % 2 == 0:
            return -torch.sum(torch.sin(x) * (torch.sinh(x**2/math.pi)**(2*m)))
        else:
            return torch.sum(torch.sin(x) * (torch.sinh(x**2/math.pi)**(2*m)))


class Eggholder(object):
    """ 
    Global minima:
        x*=(512, 404.2319)
    """

    def __call__(self, x):
        assert x.shape[0] == 2
        return -(x[1]+47)*torch.sin(torch.sqrt(torch.abs(x[1]+x[0]/2+47)))-x[0]*torch.sin(torch.sqrt(torch.abs(x[0]-(x[1]+47))))

class Ackley(object):
    """ 
    Global minima:
        x*=[0]^n
    """

    def __call__(self, x, a=20, b=0.2, c=2*math.pi):
        return -a*torch.exp(-b*torch.sqrt(torch.mean(x**2)) - torch.exp(torch.mean(torch.cos(c*x)))) + a + math.e


"""
Below are benchmark funs from http://benchmarkfcns.xyz/fcns that I haven't 
seen used in a Bayes-opt paper
"""

class Ackley3(object):
    """ 
    Normally evaluated on [-32, 32]

    Global minima:
        x*=( ± 0.682584587365898, -0.36075325513719)

    """

    def __call__(self, x):
        return -200*torch.exp(-0.2*torch.sqrt(torch.sum(x[:2]**2))) + 5*torch.exp(torch.cos(3*x[0])+torch.sin(3*x[1]))


class Ackley4(object):
    """ 
    Normally evaluated on [-35, 35]^d

    Global minima:
        d=2 at x*=( -1.51, -0.755)

    """

    def __call__(self, x):
        xsq = x**2
        return torch.sum(math.e**(-0.2)*torch.sqrt(xsq[:-1]+xsq[1:]) + 3*(torch.cos(2*x[:-1]) + torch.sin(2*x[1:])))


class Adjiman(object):
    """ 
    Normally evaluated on x[0] \in [-1, 2] and x[1] \in [-1, 1]

    Global minima:
        on x\in [-1, 2]x*nd x\in [-1, 1] cube, x*=(0, 0)

    """

    def __call__(self, x):
        return torch.cos(x[0])*torch.sin(x[1]) - x[0]/(x[1]**2+1)


class Apline1(object):
    """ 
    Normally evaluated on [0, 10]^d 

    Global minima:
        x*=(0, 0)

    """

    def __call__(self, x):
        return torch.sum(torch.abs(x*torch.sin(x)+0.1*x))


class Alpine2(object):
    """ 
    Normally evaluated on [0, 10]^d

    Global minima:
        x*=[7.917]^d

    """

    def __call__(self, x):
        return torch.prod(torch.sqrt(x)*torch.sin(x))


class Rastrigin(object):
    """ 
    Normally evaluated on [-5.12, 5.12]

    Global minima:
        x*=[0]^d
    """

    def __call__(self, x):
        d = x.shape[0]
        return 10*d + torch.sum(x**2-10*torch.cos(2*math.pi*x))

class Rosenbrock(object):
    """ 
    Normally evaluated on [-5, 10]^d

    Global minima:
        x*=[1]^d
    """

    def __call__(self, x, a=1, b=100):
        return torch.sum(b*(x[1:]-x[:-1]**2)**2 + (x[:-1]-a)**2)


class CrossInTray(object):
    """ 
    Normally evaluated on [-10, 10]^2

    Global minima:
        x*=( ± 1.3491, -± 1.3491 )
    """

    def __call__(self, x):
        return -0.0001*(
                torch.abs(
                    torch.prod(torch.sin(x[:2]))*
                    torch.exp(
                        torch.abs(100 - torch.sqrt(torch.sum(x[:2]**2))/math.pi))+1)+1)**0.1
