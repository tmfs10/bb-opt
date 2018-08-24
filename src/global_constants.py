
import torch

_eps = 1.0e-5

dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
ByteTensor = torch.cuda.ByteTensor if torch.cuda.is_available() else torch.ByteTensor
LongTensor = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor
