#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os

if not os.path.exists('./data/'):
    os.makedirs('./data/')

PATH = '../../input/'
train_data_path = PATH + 'train_data.csv'
test_data_path = PATH + 'test_data.csv'

train_data_bpe_path = PATH + 'train_data_bpe.csv'
test_data_bpe_path = PATH + 'test_data_bpe.csv'


def concat(train, test, save_name):
    data = []
    with open(train, 'r') as f:
        for line in f.readlines():
            data.append(line)

    with open(test, 'r') as f:
        for line in f.readlines():
            data.append(line)

    with open('./data/{}.csv'.format(save_name), 'w') as f:
        for line in data:
            f.writelines(line)


concat(train_data_path, test_data_path, 'data')
concat(train_data_bpe_path, test_data_bpe_path, 'data_bpe')
