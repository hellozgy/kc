#!/usr/bin/env python
# -*- coding: utf-8 -*-
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""实现出多标签的分层划分方法
过程写的太乱，
思路是计算每个label应该抽取的数目label_size,
计算test, train, val中该标签应该抽取多少个，
然后减去已经抽到的该标签的个数，
之后，从剩余未抽样的样本中抽取出满足需求的样本
重复进行，直到抽取到最后一个label。
"""
import pandas as pd
import numpy as np


data = pd.read_csv('../train.csv')

# 整个过程中，我要维持三张表
data_index = np.arange(len(data))
test_index = []
val_index = []
train_index = []

data_size = len(data)
train_size = len(data) - 20000
test_size = 10000
val_size = 10000


def get_label_index(df, col, cur_data_index):
    """计算现有数据中某个标签为1的index"""
    return cur_data_index[df.loc[cur_data_index, col] == 1]


def sample(col, col_index, test_col_size, val_col_size):
    """采样的过程，输入某个label要采样的数据，更新test index， val index, train index
    Args:
         col: label名
         col_index: 这个label  = 1的index
         test_col_size: 抽样的test大小
         val_col_size: 抽样的val大小
    """
    print("{} 的test要采样{}个， val要采样{}个， train要采样{}个".format(col, test_col_size, val_col_size,
                                                          len(col_index) - test_col_size - val_col_size))
    test_col_index = np.random.choice(col_index, size=test_col_size, replace=False)

    for v in test_col_index:
        test_index.append(v)
    # 去除test已经抽样的部分
    print('before delete', len(col_index))
    col_index = np.setdiff1d(col_index, test_col_index)
    print('after delete', len(col_index))
    val_col_index = np.random.choice(col_index, size=val_col_size, replace=False)

    for v in val_col_index:
        val_index.append(v)

    # 去除val已经抽样的部分，这部分就是train的部分
    train_col_index = np.setdiff1d(col_index, val_col_index)
    for v in train_col_index:
        train_index.append(v)

    print('经过{}的抽样，目前得到test的size是{}， val的size是{}， train的size是{}'.format(col, len(test_index),
                                                                        len(val_index), len(train_index)))

# 第一步，分类threat标签
col = 'threat'
threat_size = np.sum(data[col] == 1)

print('threat size', threat_size)

test_threat_size = val_threat_size = int(threat_size * test_size / data_size)

print('要抽取的test size大小为', test_threat_size)

# 计算现有的数据中的threat的index
threat_index = get_label_index(data, col, data_index)
assert threat_size == len(threat_index)

# 在现有的data index中去除这部分，防止出现重复采样情况
data_index = np.setdiff1d(data_index, threat_index)

sample(col, threat_index, test_threat_size, val_threat_size)


#####################################################
# 第一个抽样完成，下面进行identity hate的抽样，抽样的时候注意之前抽样重复的出情况
def calculte_have_sample_count(df, col, index):
    """用于计算已经采样的个数"""
    return np.sum(df.loc[index, col] == 1)


col = 'identity_hate'
hate_index = data_index[data.loc[data_index, col] == 1]
# 数据集种hate size
hate_size = np.sum(data[col] == 1)

# data index中去除这部分，防止重复采样
data_index = np.setdiff1d(data_index, hate_index)

# 考虑hate test, 首先采样的个数要减去已经采样的个数，然后再进行采样
test_hate_size = int(hate_size * test_size / data_size) - calculte_have_sample_count(data, col, test_index)
print('要采样的hate test size是', test_hate_size)

val_hate_size = int(hate_size * val_size / data_size) - calculte_have_sample_count(data, col, val_index)
print('要采用的hate val size是', val_hate_size)

sample(col, hate_index, test_hate_size, val_hate_size)

###########################################
# 开始计算identity_hate
col = 'severe_toxic'
severe_toxic_size = np.sum(data[col] == 1)

severe_toxic_index = data_index[data.loc[data_index, col] == 1]

# 更新data index
data_index = np.setdiff1d(data_index, severe_toxic_index)

severe_toxic_test_size = int(severe_toxic_size * test_size / data_size) - calculte_have_sample_count(data, col, test_index)
severe_toxic_val_size = int(severe_toxic_size * val_size / data_size) - calculte_have_sample_count(data, col, val_index)

sample(col, severe_toxic_index, severe_toxic_test_size, severe_toxic_val_size)

###############################################
# insult
col = 'insult'
insult_size = np.sum(data[col] == 1)
insult_index = data_index[data.loc[data_index, col] == 1]
# 更新data index
data_index = np.setdiff1d(data_index, insult_index)
# 计算insult test size
insult_test_size = int(insult_size * test_size / data_size) - calculte_have_sample_count(data, col, test_index)
# 计算insult val size
insult_val_size = int(insult_size * val_size / data_size) - calculte_have_sample_count(data, col, val_index)

sample(col, insult_index, insult_test_size, insult_val_size)

#####################################################
# obscene
col = 'obscene'
obscene_size = np.sum(data[col] == 1)
obscene_index = data_index[data.loc[data_index, col] == 1]
# 更新data idnex
data_index = np.setdiff1d(data_index, obscene_index)
# 计算obscene test size
obscene_test_size = int(obscene_size * test_size / data_size) - calculte_have_sample_count(data, col, test_index)
# 计算obscene val size
obscene_val_size = int(obscene_size * val_size / data_size) - calculte_have_sample_count(data, col, val_index)
# 开始采样
sample(col, obscene_index, obscene_test_size, obscene_val_size)

#####################################################
# toxic
col = 'toxic'
toxic_size = np.sum(data[col] == 1)
toxic_index = data_index[data.loc[data_index, col] == 1]
# 更新data index
data_index = np.setdiff1d(data_index, toxic_index)
toxic_test_size = int(toxic_size * test_size / data_size) - calculte_have_sample_count(data, col, test_index)
toxic_val_size = int(toxic_size * val_size / data_size) - calculte_have_sample_count(data, col, val_index)

sample(col, toxic_index, toxic_test_size, toxic_val_size)

print('剩下的data size是', len(data_index))

clean_test_size = test_size - len(test_index)
clean_val_size = val_size - len(val_index)

# 开始进行test的采样
clean_test_index = np.random.choice(data_index, clean_test_size, replace=False)
# 更新data index
data_index = np.setdiff1d(data_index, clean_test_index)

for v in clean_test_index:
    test_index.append(v)

# 开始进行val的采样
clean_val_index = np.random.choice(data_index, clean_val_size, replace=False)
data_index = np.setdiff1d(data_index, clean_val_index)

for v in clean_val_index:
    val_index.append(v)

for v in data_index:
    train_index.append(v)

print('train size', len(train_index))
print('test size', len(test_index))
print('val size', len(val_index))

# 以上的步骤，得到了test数据的index，train数据的index，val数据的index
# 然后就可以讲数据进行划分了


def stratified_split_data(dataset):
    """
    按照上面步骤计算得到的index，讲数据集进行划分
    :param train: 数据格式要求是numpy格式
    :return: (train data, test data, val data)
    """
    # 转化成numpy格式
    if not isinstance(dataset, np.ndarray):
        dataset = np.asarray(dataset)
    train_data = dataset[train_index]
    test_data = dataset[test_index]
    val_data = dataset[val_index]

    # 返回结果
    return train_data, test_data, val_data
