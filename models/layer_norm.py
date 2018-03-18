import torch
from torch import nn

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-5, affine=True, dim=-1):
        super(LayerNorm, self).__init__()
        self.affine = affine
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps
        self.dim = dim

    def forward(self, x):
        if self.dim!=-1 and self.dim!=len(x.size())-1:x = x.transpose(self.dim, -1)
        mean = x.mean(dim=-1, keepdim=True)
        std = (x.var(dim=-1, keepdim=True) + self.eps).sqrt()
        xx = (x - mean) / std * self.gamma + self.beta
        if self.dim != -1 and self.dim != len(x.size()) - 1: xx = xx.transpose(self.dim, -1)
        return xx