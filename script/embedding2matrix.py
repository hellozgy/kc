#coding:utf8
'''
将embedding 转成numpy矩阵
'''

import word2vec
import numpy as np

path = os.path.abspath(os.path.dirname(__file__)+'/../input/')

def main(embedding_file, vec_file):
    '''
    embedding ->numpy
    '''
    em = word2vec.load(embedding_file)
    vec = (em.vectors)
    word2id = em.vocab_hash
    np.savez_compressed(vec_file,vector=vec,word2id=word2id)

if __name__ == '__main__':
    import fire
    fire.Fire()