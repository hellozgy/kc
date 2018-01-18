#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch as t
import torch
import numpy as np
from torch import nn
from collections import OrderedDict
from .BasicModule import BasicModule


class FastText(BasicModule):
    """pytorch实现的简单fasttext分类"""
    def __init__(self, opt):
        super(FastText, self).__init__(opt)
        self.opt = opt


