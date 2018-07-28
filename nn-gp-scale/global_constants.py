
import torch as th

dtype = th.cuda.FloatTensor if th.cuda.is_available() else th.FloatTensor
FloatTensor = th.cuda.FloatTensor if th.cuda.is_available() else th.FloatTensor
ByteTensor = th.cuda.ByteTensor if th.cuda.is_available() else th.ByteTensor
LongTensor = th.cuda.LongTensor if th.cuda.is_available() else th.LongTensor
