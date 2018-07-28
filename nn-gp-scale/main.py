
import torch as th

"""

Model is the following :-

x → embedding → f(x), p(x) for being in a bin (variational if multilevel draw)
k'(x, x') = k(f(x), f(x')) * p(x) * p(x')

"""
