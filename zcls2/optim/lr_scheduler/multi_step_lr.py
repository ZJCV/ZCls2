# -*- coding: utf-8 -*-

"""
@date: 2022/4/4 下午10:42
@file: multi_step_lr.py
@author: zj
@description: 
"""

import torch.optim as optim
from torch.optim.optimizer import Optimizer


def build_multistep_lr(optimizer, milestones=None, gamma=0.1):
    assert isinstance(optimizer, Optimizer)
    if milestones is None:
        milestones = [30, 60, 80]

    return optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
