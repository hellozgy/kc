#coding:utf-8
import os
from config import opt
from dataset import KCDataset
from dataset import Constants
import models
import torch
from torch.autograd import Variable
from torch.utils import data
import tqdm
import shutil
import ipdb

torch.manual_seed(1)
torch.cuda.manual_seed(1)

def train(**kwargs):
    opt.parse(kwargs)
    save2path = './checkpoints/{}/'.format(opt.model)
    if not os.path.exists(save2path): os.system('mkdir {}'.format(save2path))
    assert opt.ngpu >= 0
    train_data = KCDataset('train', opt.max_len)
    model = getattr(models, opt.model)(opt)

if __name__ == '__main__':
    import fire
    fire.Fire()