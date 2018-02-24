# coding: utf-8
# pytorch auc

from sklearn.metrics import auc
import torch
import numpy as np


def pytorch_auc(predict, label):
    if torch.cuda.is_available():
        predict = predict.data.cup().numpy()
        label = label.data.cup().numpy()
    else:
        predict = predict.data.numpy()
        label = label.data.numpy()
    result = []
    for i in range(predict.shape[1]):
        result.append(auc(label[:, i], predict[:, i]))
    average = sum(result) / len(result)
    return average, np.array(result)

