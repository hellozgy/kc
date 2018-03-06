import torch
from torch import nn

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.affine = affine
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        # std = ((x - mean).pow(2).mean(-1, keepdim=True) + self.eps).sqrt()
        std = (x.var(dim=-1, keepdim=True) + self.eps).sqrt()
        xx = (x - mean) / std * self.gamma + self.beta
        return xx