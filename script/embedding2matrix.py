#coding:utf8
'''
将embedding 转成numpy矩阵
'''
import word2vec
import numpy as np
import sys,os,re
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
        label= np.loadtxt(os.path.join(base_dir, label_file), dtype=np.str, delimiter=',')
        docs[tag] = (data, label)
    np.savez_compressed(os.path.join(base_dir, res_file), docs=docs, vocab_size=len(word2id), word2id=word2id)


def build_glove_vec(glove, model, output_vec):
    '''
    创建语料glove词向量
    :param glove: 下载的预训练的glove词向量
    :param model: fasttext训练的词向量的二进制文件
    :param output_vec: 比赛语料的词向量
    :return:
    '''
    from fastText import load_model
    from fastText import util
    fmodel = load_model(model)
    words_list = fmodel.get_words()
    words = set(words_list)

    glove_vec = {}
    with open(glove, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip().lower().split()
            if line[0] in words:
                glove_vec[line[0]] = ' '.join(line[1:])
                words.remove(line[0])

    vectors = np.zeros((len(words_list), fmodel.get_dimension()), dtype=float)
    with open('../outdata/outvocab.txt', 'w', encoding='utf-8') as ff:
        for w in words:
            ff.write(w+'\n')

    for i in range(len(words_list)):
        wv = fmodel.get_word_vector(words_list[i])
        wv = wv / np.linalg.norm(wv)
        vectors[i] = wv

    banset = list(map(lambda x: words_list.index(x), words))
    cossims = np.zeros(len(vectors), dtype=float)
    for w in words:
        query = fmodel.get_word_vector(w)
        query = query / (np.linalg.norm(query))
        glove_vec[w] = glove_vec[words_list[util.find_nearest_neighbor(query=query, vectors=vectors, ban_set=banset, cossims=cossims)]]
    with open(output_vec, 'w', encoding='utf-8') as fw:
        fw.write('{} {}\n'.format(len(glove_vec), len(glove_vec['you'].split())))
        for w in words_list:
            fw.write(w + ' ' + glove_vec[w] + '\n')

if __name__ == '__main__':
    import fire
    fire.Fire()
    '''
    python embedding2matrix.py embed2vec vec_fasttext_bpe.txt vec_fasttext_bpe.npz
    python embedding2matrix.py word2id vec_fasttext_bpe.npz '["train_data_bpe_train.csv","train_data_bpe_val.csv","train_data_bpe_test.csv","test_data_bpe.csv"]' \
    '["train_label_bpe_train.csv","train_label_bpe_val.csv","train_label_bpe_test.csv","test_label.csv"]' '["train","val","test","commit"]' docs_bpe.npz
    
    python embedding2matrix.py build_glove_vec --glove='../outdata/glove.840B.300d.txt' --model='../input/fasttext_bpe.bin' --output_vec='../input/vec_glove_bpe.txt'
    python embedding2matrix.py embed2vec vec_glove_bpe.txt vec_glove_bpe.npz
    python embedding2matrix.py word2id vec_glove_bpe.npz '["train_data_bpe_train.csv","train_data_bpe_val.csv","train_data_bpe_test.csv","test_data_bpe.csv"]' \
    '["train_label_bpe_train.csv","train_label_bpe_val.csv","train_label_bpe_test.csv","test_label.csv"]' '["train","val","test","commit"]' glove_docs_bpe.npz
    '''