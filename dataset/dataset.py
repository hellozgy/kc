from torch.utils import data
import os,re,random
import word2vec
import ipdb
import numpy as np
from dataset.Constants import PAD_INDEX, UNK_INDEX, PAD_WORD, UNK_WORD
import datetime

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
        self.split_sentence = split_sentence
        npdata = np.load(os.path.join(base_dir, file))
        self.vocab_size = int(npdata['vocab_size'])
        docs = npdata['docs'].item()
        labels = np.row_stack([docs[tag][1] for tag in tags]).squeeze()
        self.ids = labels[:, 0] if 'commit' not in tags else labels
        self.labels = labels[:, 1:].astype(np.int) if 'commit' not in tags else np.zeros((labels.shape[0], 6))
        datas = [docs[tag][0] for tag in tags]
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
        d = d.tolist()
        index = d.index(PAD_INDEX) if PAD_INDEX in d else len(d)
        d = np.random.permutation(d[:index]).tolist()+d[index:]
        return np.asarray(d)

    def dropout(self, d, p):
        len_ = len(d)
        index = np.random.choice(len_, int(len_ * p))
        d[index] = PAD_INDEX
        return d

    def __getitem__(self, index):
        content, label, id = self.datas[index], self.labels[index], self.ids[index]
        if self.split_sentence:length = 0
        else:
            length = np.where(self.datas[index]==PAD_INDEX)[0]
            length = length[0] if len(length)>0 else len(self.datas[index])

        if self.training:
            if np.random.random() > 0.5:
                content = self.dropout(content, p=self.dropout_p)
            else:
                content = self.shuffle(content)
        return content, label, id, np.asarray([length])

    def __len__(self):
        return self.datas.shape[0]

class KCDataset10fold(data.Dataset):
    def __init__(self, index=8, max_len=100, vec_name='../input/vec_fasttext_bpe_300.txt',training=True, dropout_data=0.3,
                 split_sentence=False, max_sentence=10, max_word_persentence=30):
        '''
        :param file:the file of all data
        :param tags: a list of train,val,test,commit
        :param max_len:
        :param split_sentence:
        :param max_sentence:
        :param max_word_persentence:
        '''

        def read_data(fname):
            res = []
            with open(fname, 'r', encoding='utf-8') as f:
                for line in f:
                    res.append(line.strip())
            return res

        self.training = training
        self.dropout_data = dropout_data

        em = word2vec.load(vec_name)
        if not os.path.exists(vec_name[:-4]):
            vec = em.vectors
            pad_unk = np.zeros((2, vec.shape[1]))
            vec = np.row_stack((pad_unk, vec))
            np.savez_compressed(vec_name[:-4]+'.npz', vec=vec)
        word2id = {k: v + 2 for k, v in em.vocab_hash.items()}
        word2id[PAD_WORD] = PAD_INDEX
        word2id[UNK_WORD] = UNK_INDEX
        self.vocab_size = len(word2id)

        self.train = read_data(os.path.join(base_dir, '../tenfold/train_data_train_{}.csv'.format(index)))
        self.train_label = read_data(os.path.join(base_dir, '../tenfold/train_label_train_{}.csv'.format(index)))
        assert len(self.train)==len(self.train_label)
        self.train = [[word2id.get(word, UNK_INDEX) for word in line.split()[:max_len]] for line in self.train]
        self.train = [line+[PAD_INDEX]*(max_len-len(line)) for line in self.train]
        self.train = np.asarray(self.train)
        self.train_label = [[int(t) for t in line.split(',')[1:]] for line in self.train_label]
        self.train_label = np.asarray(self.train_label)

        self.val = read_data(os.path.join(base_dir, '../tenfold/train_data_test_{}.csv'.format(index)))
        self.val_label = read_data(os.path.join(base_dir, '../tenfold/train_label_test_{}.csv'.format(index)))
        assert len(self.val) == len(self.val_label)
        self.val = [[word2id.get(word, UNK_INDEX) for word in line.split()[:max_len]] for line in self.val]
        self.val = [line + [PAD_INDEX] * (max_len - len(line)) for line in self.val]
        self.val = np.asarray(self.val)
        self.val_label = [[int(t) for t in line.split(',')[1:]] for line in self.val_label]
        self.val_label = np.asarray(self.val_label)


        self.data = self.train if self.training else self.val
        self.label = self.train_label if self.training else self.val_label

    def set_train(self, train=True):
        self.training = train
        if train:
            self.data = self.train
            self.label = self.train_label
        else:
            self.data = self.val
            self.label = self.val_label

    def shuffle(self, d):
        d = d.tolist()
        index = d.index(PAD_INDEX) if PAD_INDEX in d else len(d)
        d = np.random.permutation(d[:index]).tolist()+d[index:]
        return np.asarray(d)

    def dropout(self, d, p):
        len_ = len(d)
        index = np.random.choice(len_, int(len_ * p))
        d[index] = PAD_INDEX
        return d

    def __getitem__(self, index):
        data, label = self.data[index], self.label[index]

        if self.training:
            if np.random.random() > 0.5:
                data = self.dropout(data, p=self.dropout_data)
            else:
                data = self.shuffle(data)
        return data, label

    def __len__(self):
        return self.data.shape[0]

class KCDatasetTest(data.Dataset):
    def __init__(self, data_file, label_file,  max_len=100, vec_name='./input/vec_fasttext_bpe_300.txt',
                 split_sentence=False, max_sentence=10, max_word_persentence=30):
        '''
        :param file:the file of all data
        :param tags: a list of train,val,test,commit
        :param max_len:
        :param split_sentence:
        :param max_sentence:
        :param max_word_persentence:
        '''

        def read_data(fname):
            res = []
            with open(fname, 'r', encoding='utf-8') as f:
                for line in f:
                    res.append(line.strip())
            return res


        em = word2vec.load(vec_name)
        if not os.path.exists(vec_name[:-4]):
            vec = em.vectors
            pad_unk = np.zeros((2, vec.shape[1]))
            vec = np.row_stack((pad_unk, vec))
            np.savez_compressed(vec_name[:-4]+'.npz', vec=vec)
        word2id = {k: v + 2 for k, v in em.vocab_hash.items()}
        word2id[PAD_WORD] = PAD_INDEX
        word2id[UNK_WORD] = UNK_INDEX
        self.vocab_size = len(word2id)

        self.data = read_data(data_file)
        self.data_label = read_data(label_file)
        assert len(self.train)==len(self.train_label)
        self.data = [[word2id.get(word, UNK_INDEX) for word in line.split()[:max_len]] for line in self.data]
        self.data = [line+[PAD_INDEX]*(max_len-len(line)) for line in self.data]
        self.data = np.asarray(self.data)
        self.data_id = [[line.split(',')[0]] for line in self.data_label]
        self.data_label = [[int(t) for t in line.split(',')[1:]] if len(line.split(','))==7 else [0]*6 for line in self.data_label]
        self.data_label = np.asarray(self.data_label)

    def __getitem__(self, index):
        return self.data[index], self.data_label[index], self.data_id[index]

    def __len__(self):
        return self.data.shape[0]


if __name__ == '__main__':
    ds = KCDataset('docs_bpe.npz', ['test'], max_len=100, split_sentence=True)
    for k,v in ds:
        ipdb.set_trace()
        print(k);print(v)
