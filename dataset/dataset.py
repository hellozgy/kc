from torch.utils import data
import os, re
import word2vec
import ipdb
import numpy as np
import copy
from dataset.Constants import PAD_INDEX, UNK_INDEX, PAD_WORD, UNK_WORD

base_dir = os.path.abspath(os.path.dirname(__file__) + './../input/')

class HANDataset(data.Dataset):
    def __init__(self, index = 8, max_len=200, vec_name = '../input/vec_fasttext_bpe_300.txt',
                 training = True, dropout_data = 0.3, bpe = True, max_sentence = 20,  max_word_persentence = 50):
        def read_data(fname):
            res = []
            with open(fname, 'r', encoding='utf-8') as f:
                for line in f:
                    res.append(line.strip())
            return res

        def get_data(datas):
            p = r'|'.join([fr'\b{word2id[punc]}\b' for punc in '-,:;.?!='])
            datas = [' '.join([str(word2id.get(word, UNK_INDEX)) for word in line.split()]) for line in datas]
            datas = [[line.strip() for line in re.split(p, line) if line.strip()!=''] for line in datas]
            datas = [[sent.split()[:max_word_persentence] for sent in doc][:max_sentence] for doc in datas]
            datas = [[[UNK_INDEX]] if len(doc)==0 else doc for doc in datas]
            datas = [doc + [[PAD_INDEX]] * (max_sentence - len(doc)) for doc in datas]
            datas = [[list(map(int, sent)) + [PAD_INDEX] * (max_word_persentence - len(sent)) for sent in doc] for doc in datas]
            for index in range(len(datas)):
                d = datas[index]
                if d[0][0]==0:
                    ipdb.set_trace()
            datas = np.asarray(datas)
            return datas


        self.training = training
        self.dropout_data = dropout_data

        em = word2vec.load(vec_name)
        word2id = {k: v + 2 for k, v in em.vocab_hash.items()}
        word2id[PAD_WORD] = PAD_INDEX
        word2id[UNK_WORD] = UNK_INDEX
        self.vocab_size = len(word2id)
        if not os.path.exists(vec_name[:-4]):
            vec = em.vectors
            pad_unk = np.zeros((2, vec.shape[1]))
            vec = np.row_stack((pad_unk, vec))

            category = [['toxic', PAD_WORD], ['severe','toxic'],['obscene', PAD_WORD],['threat', PAD_WORD],['insult', PAD_WORD],['identity','hate']]
            category = np.asarray([[word2id.get(cc) for cc in c] for c in category])
            np.savez_compressed(vec_name[:-4]+'.npz', vec=vec, category=category)



        self.train = read_data(os.path.join(base_dir, '../tenfold/train_data{}_train_{}.csv'.format(('_bpe' if bpe else ''), index)))
        self.train_label = read_data(os.path.join(base_dir, '../tenfold/train_label_train_{}.csv'.format(index)))
        self.train_label = [[int(t) for t in line.split(',')[1:]] for line in self.train_label]
        self.train_label = np.asarray(self.train_label)
        self.val = read_data(os.path.join(base_dir, '../tenfold/train_data{}_test_{}.csv'.format(('_bpe' if bpe else ''), index)))
        self.val_label = read_data(os.path.join(base_dir, '../tenfold/train_label_test_{}.csv'.format(index)))
        self.val_label = [[int(t) for t in line.split(',')[1:]] for line in self.val_label]
        self.val_label = np.asarray(self.val_label)

        self.train = get_data(self.train)
        self.val = get_data(self.val)

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

    def dropout(self, dd, p):
        for d in dd:
            x = d
            d = d.tolist()
            length = d.index(PAD_INDEX) if PAD_INDEX in d else len(d)
            if length==0:continue
            _index = np.random.choice(length, int(length * p))
            if len(_index.tolist())>0:
                x[_index] = PAD_INDEX
        return dd

    def __getitem__(self, index):
        data, label, bw, length = self.data[index], self.label[index], np.asarray([1]), np.asarray([1])

        if self.training:
            if np.random.random() > 0.5:
                data = self.dropout(data, p=self.dropout_data)
            else:
                data = self.shuffle(data)
        return data, label, bw, length

    def __len__(self):
        return self.data.shape[0]

class KCDataset10fold(data.Dataset):
    def __init__(self, index=8, max_len=100, vec_name='../input/vec_fasttext_bpe_300.txt',training=True, dropout_data=0.3, bpe=True,
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

        def get_badword():
            bw = set()
            with open('./input/badword.txt', 'r', encoding='utf-8') as f:
                for line in f:
                    bw.add(line.strip())
            return bw

        self.training = training
        self.dropout_data = dropout_data
        self.max_len = max_len

        em = word2vec.load(vec_name)
        word2id = {k: v + 2 for k, v in em.vocab_hash.items()}
        word2id[PAD_WORD] = PAD_INDEX
        word2id[UNK_WORD] = UNK_INDEX
        self.vocab_size = len(word2id)
        if not os.path.exists(vec_name[:-4]):
            vec = em.vectors
            pad_unk = np.zeros((2, vec.shape[1]))
            vec = np.row_stack((pad_unk, vec))

            category = [['toxic', PAD_WORD], ['severe','toxic'],['obscene', PAD_WORD],['threat', PAD_WORD],['insult', PAD_WORD],['identity','hate']]
            category = np.asarray([[word2id.get(cc) for cc in c] for c in category])
            np.savez_compressed(vec_name[:-4]+'.npz', vec=vec, category=category)


        bw = get_badword()
        bw = set([word2id[w] for w in bw if w in word2id])

        self.train = read_data(os.path.join(base_dir, '../tenfold/train_data{}_train_{}.csv'.format(('_bpe' if bpe else ''), index)))
        self.train_label = read_data(os.path.join(base_dir, '../tenfold/train_label_train_{}.csv'.format(index)))

        # n = len(self.train)
        # for i in range(n-1, -1, -1):
        #     lab = self.train_label[i]
        #     if '1' not in lab and np.random.random()>0.9:
        #         self.train.pop(i)
        #         self.train_label.pop(i)


        # outdata = read_data('./input/outdata_data_bpe_final.csv')
        # outdata_label = read_data('./input/outdata_label_final.csv')
        # self.train.extend(outdata)
        # self.train_label.extend(['ididididid,'+tt for tt in outdata_label])

        assert len(self.train)==len(self.train_label)
        self.train = [[word2id.get(word, UNK_INDEX) for word in line.split()[:max_len]] for line in self.train]
        self.train_len = [[len(line)] for line in self.train]
        self.train = [line+[PAD_INDEX]*(max_len-len(line)) for line in self.train]
        self.train_label = [[int(t) for t in line.split(',')[1:]] for line in self.train_label]
        self.train_bw = [[w for w in line if w in bw][:10] for line in self.train]
        self.train_bw = [line+[PAD_INDEX]*(10-len(line)) for line in self.train_bw]


        self.train = np.asarray(self.train)
        self.train_bw = np.asarray(self.train_bw)
        self.train_label = np.asarray(self.train_label)
        self.train_len = np.asarray(self.train_len)

        self.val = read_data(os.path.join(base_dir, '../tenfold/train_data{}_test_{}.csv'.format(('_bpe' if bpe else ''),index)))
        self.val_label = read_data(os.path.join(base_dir, '../tenfold/train_label_test_{}.csv'.format(index)))
        assert len(self.val) == len(self.val_label)
        self.val = [[word2id.get(word, UNK_INDEX) for word in line.split()[:max_len]] for line in self.val]
        self.val_len = np.asarray([[len(line)] for line in self.val])
        self.val = [line + [PAD_INDEX] * (max_len - len(line)) for line in self.val]
        self.val_label = [[int(t) for t in line.split(',')[1:]] for line in self.val_label]
        self.val_bw = [[w for w in line if w in bw][:10] for line in self.val]
        self.val_bw = [line + [PAD_INDEX] * (10 - len(line)) for line in self.val_bw]
        self.val = np.asarray(self.val)
        self.val_bw = np.asarray(self.val_bw)
        self.val_label = np.asarray(self.val_label)


        self.data = self.train if self.training else self.val
        self.data_len = self.train_len if self.training else self.val_len
        self.data_bw = self.train_bw if self.training else self.val_bw
        self.label = self.train_label if self.training else self.val_label

    def set_train(self, train=True):
        self.training = train
        if train:
            self.data = self.train
            self.label = self.train_label
            self.data_bw = self.train_bw
            self.data_len = self.train_len
        else:
            self.data = self.val
            self.label = self.val_label
            self.data_bw = self.val_bw
            self.data_len = self.val_len

    def shuffle(self, d):
        d = d.tolist()
        index = d.index(PAD_INDEX) if PAD_INDEX in d else len(d)
        d = np.random.permutation(d[:index]).tolist()+d[index:]
        return np.asarray(d)

    def dropout(self, d, p):
        d = copy.deepcopy(d)
        len_ = len(d)
        index = np.random.choice(len_, int(len_ * p))
        d[index] = PAD_INDEX
        return d

    def __getitem__(self, index):
        data, label, bw, length = self.data[index], self.label[index], self.data_bw[index], self.data_len[index]

        if self.training:
            if np.random.random() > 0.5:
                data = self.dropout(data, p=self.dropout_data)
            else:
                data = self.shuffle(data)
        return data, label, bw, np.asarray([length])

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
        self.data = [[word2id.get(word, UNK_INDEX) for word in line.split()[:max_len]] for line in self.data]
        self.data = [line+[PAD_INDEX]*(max_len-len(line)) for line in self.data]
        self.data = np.asarray(self.data)
        self.data_id = [line.split(',')[0] for line in self.data_label]
        self.data_label = [[int(t) for t in line.split(',')[1:]] if len(line.split(','))==7 else [1]*6 for line in self.data_label]
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
