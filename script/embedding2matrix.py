#coding:utf8
'''
将embedding 转成numpy矩阵
'''
import word2vec
import numpy as np
import sys, os
sys.path.append('../dataset')
import Constants

base_dir = os.path.abspath(os.path.dirname(__file__)+'./../input/')


def embed2vec(embedding_file, vec_file):
    '''
    embedding ->numpy
    '''
    em = word2vec.load(os.path.join(base_dir, embedding_file))
    vec = em.vectors
    pad_unk = np.zeros((2, vec.shape[1]))
    vec = np.row_stack((pad_unk, vec))
    word2id = {k:v+2 for k, v in em.vocab_hash.items()}
    word2id[Constants.PAD_WORD] = Constants.PAD_INDEX
    word2id[Constants.UNK_WORD] = Constants.UNK_INDEX
    np.savez_compressed(os.path.join(base_dir, vec_file),vector=vec,word2id=word2id)


def word2id(vec_file, data_files, label_files, tags, res_file):
    word2id = np.load(os.path.join(base_dir, vec_file))['word2id'].item()
    docs = {}
    for data_file, label_file, tag in zip(data_files, label_files, tags):
        data = []
        with open(os.path.join(base_dir,data_file), 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                data.append([word2id[word] for word in line.split() if word in word2id])
        label= np.loadtxt(os.path.join(base_dir, label_file), dtype=int, delimiter=',')
        docs[tag] = (data, label)
    np.savez_compressed(os.path.join(base_dir, res_file), docs=docs, vocab_size=len(word2id))

if __name__ == '__main__':
    import fire
    fire.Fire()
    '''
    python embedding2matrix.py embed2vec vec_fasttext_bpe.txt vec_fasttext_bpe.npz
    python embedding2matrix.py word2id vec_fasttext_bpe.npz '["train_data_bpe_train.csv","train_data_bpe_val.csv","train_data_bpe_test.csv","test_data_bpe.csv"]' \
    '["train_label_bpe_train.csv","train_label_bpe_val.csv","train_label_bpe_test.csv","test_label.csv"]' '["train","val","test","commit"]' docs_bpe.npz
    '''