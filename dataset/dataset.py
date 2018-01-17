from torch.utils import data
import os
import word2vec
import ipdb
import numpy as np
from dataset.Constants import PAD_INDEX

base_dir = os.path.abspath(os.path.dirname(__file__)+'./../input/')
class KCDataset(data.Dataset):
    def __init__(self, file, tag, max_len):
        '''
        :param tag: train,val,test,commit
        :param max_len:
        '''
        self.tag = tag
        self.max_len = max_len
        npdata = np.load(os.path.join(base_dir, file))
        self.vocab_size = int(npdata['vocab_size'])
        doc = npdata['docs'].item()[tag]
        self.data = np.asarray([(d + [PAD_INDEX] * (self.max_len - len(d)))[: max_len] for d in doc[0]])
        self.label = doc[1][:,1:] if self.tag!='commit' else doc[1]

    def __getitem__(self, index):
        return (self.data[index], self.label[index])

    def __len__(self):
        return self.data.shape[0]

if __name__ == '__main__':
    ds = KCDataset('docs_bpe.npz', 'train', max_len=100)
    for k,v in ds:
        print(k);print(v)
