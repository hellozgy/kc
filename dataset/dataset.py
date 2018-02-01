from torch.utils import data
import os,re,random
import word2vec
import ipdb
import numpy as np
from dataset.Constants import PAD_INDEX

base_dir = os.path.abspath(os.path.dirname(__file__) + './../input/')

class KCDataset(data.Dataset):
    def __init__(self, file, tags, max_len, split_sentence=False, max_sentence=10, max_word_persentence=30, training=False, dropout_data=0.5):
        '''
        :param file:the file of all data
        :param tags: a list of train,val,test,commit
        :param max_len:
        :param split_sentence:
        :param max_sentence:
        :param max_word_persentence:
        '''
        self.tags = tags
        self.max_len = max_len
        self.training = training
        self.dropout_p = dropout_data
        npdata = np.load(os.path.join(base_dir, file))
        self.vocab_size = int(npdata['vocab_size'])
        labels = np.row_stack([npdata['docs'].item()[tag][1] for tag in tags]).squeeze()
        self.ids = labels[:, 0] if 'commit' not in tags else labels
        self.labels = labels[:, 1:].astype(np.int) if 'commit' not in tags else np.zeros((labels.shape[0], 6))
        datas = [npdata['docs'].item()[tag][0] for tag in tags]
        if not split_sentence:
            self.datas = np.asarray([(d + [PAD_INDEX] * (self.max_len - len(d)))[: max_len] for data in datas for d in data])
        else:
            word2id = npdata['word2id'].item()
            datas = [' '.join(list(map(str, doc))) for data in datas for doc in data]
            p = r'|'.join([fr'\b{word2id[punc]}\b' for punc in '-,:;.?!'])
            datas = [re.split(p, doc) for doc in datas]
            datas = [[sent.split()[:max_word_persentence] for sent in doc if len(sent.strip())>0][:max_sentence] for doc in datas]
            datas = [doc+[[PAD_INDEX]]*(max_sentence-len(doc)) for doc in datas]
            datas = [[list(map(int, sent))+[PAD_INDEX]*(max_word_persentence-len(sent)) for sent in doc] for doc in datas]
            self.datas = np.asarray(datas)
        self.datas = self.datas.astype(np.int)

    def shuffle(self, d):
        return np.random.permutation(d.tolist())

    def dropout(self, d, p=0.5):
        len_ = len(d)
        index = np.random.choice(len_, int(len_ * p))
        d[index] = 0
        return d

    def __getitem__(self, index):
        content, label, id = self.datas[index], self.labels[index], self.ids[index]
        if self.training:
            if random.random() > 0.5:
                content = self.dropout(content, p=self.dropout_p)
            else:
                content = self.shuffle(content)
        return content, label, id

    def __len__(self):
        return self.datas.shape[0]


if __name__ == '__main__':
    ds = KCDataset('docs_bpe.npz', ['test'], max_len=100, split_sentence=True)
    for k,v in ds:
        ipdb.set_trace()
        print(k);print(v)
