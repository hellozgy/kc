from torch.utils import data
import os,re
import word2vec
import ipdb
import numpy as np
from dataset.Constants import PAD_INDEX

base_dir = os.path.abspath(os.path.dirname(__file__) + './../input/')


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

class KCDatasetSplitSentence(data.Dataset):
    def __init__(self, file, tags, max_len, split_sentence=False, max_sentence=10, max_word_persentence=30):
        '''
        :param tag: a list of train,val,test,commit
        :param max_len:
        '''
        self.tags = tags
        self.max_len = max_len
        npdata = np.load(os.path.join(base_dir, file))
        self.vocab_size = int(npdata['vocab_size'])
        labels = np.row_stack([npdata['docs'].item()[tag][1] for tag in tags])
        self.labels = labels[:, 1:] if 'commit' not in tags else labels
        datas = [npdata['docs'].item()[tag][0] for tag in tags]
        if not split_sentence:
            self.datas = np.asarray([(d + [PAD_INDEX] * (self.max_len - len(d)))[: max_len] for data in datas for d in data])
        else:
            word2id = npdata['word2id'].item()
            datas = [data.tolist() for data in datas]
            datas = [' '.join(doc) for data in datas for doc in data]
            p = '|'.join(['\b{}\b'.format(word2id[punc]) for punc in '-,:;.?!'])
            datas = [re.split(p, doc) for doc in datas]
            datas = [[sent.split()[:max_word_persentence] for sent in doc if len(sent.strip())>0][:max_sentence] for doc in datas]
            datas = [doc+[[PAD_INDEX]]*(max_sentence-len(doc)) for doc in datas]
            datas = [[sent+[PAD_INDEX](max_word_persentence-len(sent)) for sent in doc] for doc in datas]
            self.datas = np.asarray(datas)

    def __getitem__(self, index):
        return (self.datas[index], self.labels[index])

    def __len__(self):
        return self.datas.shape[0]


if __name__ == '__main__':
    ds = KCDatasetSplitSentence('docs_bpe.npz', 'train', max_len=100)
    for k,v in ds:
        print(k);print(v)
