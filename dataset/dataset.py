from torch.utils import data
import os
import word2vec
import ipdb
import numpy as np
from Constants import PAD_INDEX, UNK_INDEX

data_dir = os.path.abspath(os.path.dirname(__file__)+'/../input/')
class KCDataset(data.Dataset):
    def __init__(self, subset, max_len):
        '''
        :param subset: train,val,test,commit
        :param max_len:
        '''




    def __getitem__(self, index):
        pass

    def __len__(self):
        pass

if __name__ == '__main__':
    path = os.path.abspath(os.path.dirname(__file__)+'/../input/')
    ipdb.set_trace()
    print(path)