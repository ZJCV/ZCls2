# -*- coding: utf-8 -*-

"""
@date: 2022/4/4 上午11:17
@file: __init__.py.py
@author: zj
@description:
"""

from yacs.config import CfgNode
from zcls2.config.configs.defaults import _C
from .configs import dataset, dataloader, model, lr_scheduler, optimizer, custom_config, transform

dataloader.add_config(_C)
dataset.add_config(_C)
lr_scheduler.add_config(_C)
model.add_config(_C)
optimizer.add_config(_C)
transform.add_config(_C)

# Add custom config with default values.
custom_config.add_custom_config(_C)


def get_cfg_defaults() -> CfgNode:
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()

# cfg = get_cfg_defaults()
