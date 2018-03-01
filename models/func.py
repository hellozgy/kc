import torch.nn.functional as F
def swish(x):
    return x*F.sigmoid(x)
