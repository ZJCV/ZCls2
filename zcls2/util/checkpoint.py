# -*- coding: utf-8 -*-

"""
@date: 2022/4/3 下午1:40
@file: checkpoint.py
@author: zj
@description: 
"""

import os
import torch
import shutil

from zcls2.util import logging

logger = logging.get_logger(__name__)


def save_checkpoint(state, is_best, output_dir='outputs', filename='checkpoint.pth.tar'):
    save_path = os.path.join(output_dir, filename)
    logger.info(f"Save to {save_path}")
    torch.save(state, save_path)
    if is_best:
        best_path = os.path.join(output_dir, 'model_best.pth.tar')
        logger.info(f"Copy to {best_path}")
        shutil.copyfile(save_path, best_path)
