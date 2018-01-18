from torch.utils import data
import os
import word2vec
import ipdb
import numpy as np
from dataset.Constants import PAD_INDEX

base_dir = os.path.abspath(os.path.dirname(__file__)+'./../input/')
class KCDataset(data.Dataset):
    def __init__(self, file, tags, max_len):
        '''
        :param tag: a list of train,val,test,commit
        :param max_len:
        '''
        self.tags = tags
        self.max_len = max_len
        npdata = np.load(os.path.join(base_dir, file))
        self.vocab_size = int(npdata['vocab_size'])
        datas = [npdata['docs'].item()[tag][0] for tag in tags]
        labels = np.row_stack([npdata['docs'].item()[tag][1] for tag in tags])
        self.datas = np.asarray([(d + [PAD_INDEX] * (self.max_len - len(d)))[: max_len] for data in datas for d in data])
        self.labels = labels[:,1:] if 'commit' not in tags else labels

    def __getitem__(self, index):
        return (self.datas[index], self.labels[index])

    def __len__(self):
        return self.datas.shape[0]

if __name__ == '__main__':
    ds = KCDataset('docs_bpe.npz', 'train', max_len=100)
    for k,v in ds:
        print(k);print(v)
