
import torch
import torch.nn as nn
import numpy as np
import math

"""
values taken from 
    * https://www.sfu.ca/~ssurjano/shekel.html 
    * http://benchmarkfcns.xyz/benchmarkfcns/ackleyfcn.html 
    * wikipedia
"""

class TestFunction(object):
    def sample(self, num_samples):
        x = torch.rand((num_samples, self.ndim()))
        y = self.__call__(x)

        return x, y

class shekel(object):
    """ 
    Global minima:
        m=5,7,10, at x*=(4,4,4,4)
    """

    def __init__(self, device='cuda', m=10):
        self.beta = 0.1*torch.Tensor([1, 2, 2, 4, 4, 6, 3, 7, 5, 5.], device=device)
        self.C = torch.tensor(
                [4, 4, 4, 4],
                [1, 1, 1, 1],
                [8, 8, 8, 8],
                [6, 6, 6, 6],
                [3, 7, 3, 7],
                [2, 9, 2, 9],
                [5, 3, 5, 3],
                [8, 1, 8, 1],
                [6, 2, 6, 2],
                [7, 3.6, 7, 3.6],
                device=device)
        assert m >= 1
        assert m <= 10
        self.m = m

    def ndim(self):
        return 10

    def optimum(self):
        if self.m == 5:
            return 10.1532
        elif self.m == 7:
            return 10.4029
        elif self.m == 10:
            return 10.5364
        else:
            assert False

    def x_transform(self, x):
        return x*5+5.

    def __call__(self, x):
        x = self.x_transform(x)
        return -torch.sum((torch.sum((x-self.C[:self.m])**2, dim=1) + self.beta[:self.m])**-1)


class michalewicz(object):
    """ 
    Global minima:
        d=2, x*=(2.2, 1.57)
    """

    def __call__(self, x, m=10):
        x = self.x_transform(x)
        d = x.shape[0]
        if m % 2 == 0:
            return -torch.sum(torch.sin(x) * (torch.sinh(x**2/math.pi)**(2*m)))
        else:
            return torch.sum(torch.sin(x) * (torch.sinh(x**2/math.pi)**(2*m)))


class eggholder(TestFunction):
    """ 
    Global minima:
        x*=(512, 404.2319)
    """

    def optimum(self):
        return 959.6407

    def ndim(self):
        return 2

    def x_transform(self, x):
        return x*512.

    def __call__(self, x):
        assert x.shape[1] == 2
        x = self.x_transform(x)
        return -(-(x[:,1]+47)*torch.sin(torch.sqrt(torch.abs(x[:,1]+x[:,0]/2+47)))-x[:,0]*torch.sin(torch.sqrt(torch.abs(x[:,0]-(x[:,1]+47)))))

class ackley(TestFunction):
    """ 
    Global minima:
        x*=[0]^n
    """

    def optimum(self):
        return 0

    def ndim(self):
        return 5

    def x_transform(self, x):
        return (x+1)*32.768/2.

    def __call__(self, x, a=20, b=0.2, c=2*math.pi):
        x = self.x_transform(x)
        return -(-a*torch.exp(-b*torch.sqrt((x**2).mean(dim=-1)) - torch.exp(torch.cos(c*x).mean(dim=-1))) + a + math.e)


"""
Below are benchmark funs from http://benchmarkfcns.xyz/fcns that I haven't 
seen used in a Bayes-opt paper
"""

class ackley3(object):
    """ 
    Normally evaluated on [-32, 32]

    Global minima:
        x*=( ± 0.682584587365898, -0.36075325513719)

    """

    def __call__(self, x):
        x = x*32
        return -200*torch.exp(-0.2*torch.sqrt(torch.sum(x[:2]**2))) + 5*torch.exp(torch.cos(3*x[0])+torch.sin(3*x[1]))


class ackley4(object):
    """ 
    Normally evaluated on [-35, 35]^d

    Global minima:
        d=2 at x*=( -1.51, -0.755)

    """

    def __call__(self, x):
        x = x*35
        xsq = x**2
        return torch.sum(math.e**(-0.2)*torch.sqrt(xsq[:-1]+xsq[1:]) + 3*(torch.cos(2*x[:-1]) + torch.sin(2*x[1:])))


class adjiman(object):
    """ 
    Normally evaluated on x[0] \in [-1, 2] and x[1] \in [-1, 1]

    Global minima:
        on x\in [-1, 2]x*nd x\in [-1, 1] cube, x*=(0, 0)

    """

    def __call__(self, x):
        x[:,0] = x[:,0]*1.5+.5
        return torch.cos(x[:,0])*torch.sin(x[:,1]) - x[:,0]/(x[:,1]**2+1)


class apline1(TestFunction):
    """ 
    Normally evaluated on [0, 10]^d 

    Global minima:
        x*=(0, 0)

    """

    def x_transform(self, x):
        return x*5.+5

    def __call__(self, x):
        x = self.x_transform(x)
        return torch.sum(torch.abs(x*torch.sin(x)+0.1*x))


class alpine2(TestFunction):
    """ 
    Normally evaluated on [0, 10]^d

    Global minima:
        x*=[7.917]^d

    """

    def x_transform(self, x):
        return x*5.+5

    def __call__(self, x):
        x = self.x_transform(x)
        return torch.prod(torch.sqrt(x)*torch.sin(x))


class rastrigin(TestFunction):
    """ 
    Normally evaluated on [-5.12, 5.12]

    Global minima:
        x*=[0]^d
    """

    def x_transform(self, x):
        return 5.12*x

    def __call__(self, x):
        x = self.x_transform(x)
        d = x.shape[0]
        return 10*d + torch.sum(x**2-10*torch.cos(2*math.pi*x))

class rosenbrock(TestFunction):
    """ 
    Normally evaluated on [-5, 10]^d

    Global minima:
        x*=[1]^d
    """

    def x_transform(self, x):
        return x*7.5+2.5

    def __call__(self, x, a=1, b=100):
        x = self.x_transform(x)
        return torch.sum(b*(x[1:]-x[:-1]**2)**2 + (x[:-1]-a)**2)


class crossintray(TestFunction):
    """ 
    Normally evaluated on [-10, 10]^2

    Global minima:
        x*=( ± 1.3491, -± 1.3491 )
    """

    def __init__(self, scale=10):
        self.scale = scale

    def x_transform(self, x):
        return self.scale*x

    def __call__(self, x):
        x = self.x_transform(x)
        return -0.0001*(
                torch.abs(
                    torch.prod(torch.sin(x[:2]))*
                    torch.exp(
                        torch.abs(100 - torch.sqrt(torch.sum(x[:2]**2))/math.pi))+1)+1)**0.1


class bohachevsky(TestFunction):
    """
    Normally evaluated on [-100, 100]^2

    Global minima:
        x*=(0, 1, 2) for j=1,2,3
    """

    def __init__(self, scale=100):
        self.scale = scale

    def x_transform(self, x):
        return self.scale*x

    def ndim(self):
        return 2

    def optimum(self):
        return 0.

    def __call__(self, x):
        x = self.x_transform(x)
        return -(x[:, 0]**2 + 2*x[:, 1]**2 - 0.3*torch.cos(3*math.pi*x[:, 0]) - 0.4*torch.cos(4*math.pi*x[:, 1]) + 0.7)


class hartmann6d(TestFunction):
    """
    Normally evaluated on (0, 1)^6

    Global minima:
        x*=(0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573)
    """

    def __init__(self, device='cuda'):
        self.P = 1e-4 * torch.tensor(
                [
                    [1312, 1696, 5596, 124., 8283, 5886],
                    [2329, 4135, 8307, 3736, 1004, 9991],
                    [2348, 1451, 3522, 2883, 3047, 6650],
                    [4047, 8828, 8732, 5743, 1091, 381.],
                ],
                device=device
                )

        self.A = torch.tensor(
                [
                    [10, 3, 17, 3.5, 1.7, 8],
                    [.05, 10, 17, .1, 8, 14],
                    [3, 3.5, 1.7, 10, 17, 8],
                    [17, 8, .05, 10, .1, 14],
                ],
                device=device
                )

        self.alpha = torch.tensor([1, 1.2, 3, 3.2], device=device)

    def optimum(self):
        return 3.32237

    def ndim(self):
        return 6

    def x_transform(self, x):
        return x/2.+0.5

    def __call__(self, x):
        assert len(x.shape) == 2
        assert x.shape[1] == 6
        x = x.to(self.P.device)
        x = self.x_transform(x)
        t1 = (x.unsqueeze(1)-self.P)**2
        t2 = self.A * t1
        t3 = torch.exp(-t2.sum(dim=-1))
        t4 = self.alpha*t3
        return t4.sum(-1)

class branin(TestFunction):
    def __init__(self):
        self.a = 1
        self.b = 5.1/(4*math.pi**2)
        self.c = 5/math.pi
        self.r = 6
        self.s = 10
        self.t = 1./(8*math.pi)

    def ndim(self):
        return 2

    def optimum(self):
        return -0.397887

    def x_transform(self, x):
        x[:,0] = x[:,0]*7.5+2.5
        x[:,1] = x[:,1]*7.5+7.5
        return x

    def __call__(self, x):
        x = self.x_transform(x)

        return -(self.a * (x[:,1]-self.b*x[:,0]**2+self.c*x[:,0]-self.r)**2 + \
                self.s*(1-self.t)*torch.cos(x[:,0]) + self.s)
