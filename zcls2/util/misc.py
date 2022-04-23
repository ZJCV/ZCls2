# -*- coding: utf-8 -*-

"""
@date: 2022/4/3 下午1:41
@file: misc.py
@author: zj
@description: 
"""

import torch
import random

import numpy as np


def to_python_float(t):
    if hasattr(t, 'item'):
        return t.item()
    else:
        return t[0]


def init_seed(seed=0):
    """
    Same as Apex settings
    See
    1. [REPRODUCIBILITY](https://pytorch.org/docs/stable/notes/randomness.html)
    2. [PyTorch设置随机种子](https://blog.csdn.net/weixin_41978699/article/details/121312297)
    """
    # random.seed(seed)
    # np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
