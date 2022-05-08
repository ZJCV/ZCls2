# -*- coding: utf-8 -*-

"""
@date: 2022/4/3 下午1:39
@file: distributed.py
@author: zj
@description: 
"""

import os

import torch
import torch.distributed as dist
import torch.backends.cudnn as cudnn

from yacs.config import CfgNode
from argparse import Namespace


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


def init_dist(args: Namespace, cfg: CfgNode) -> None:
    cudnn.benchmark = False if args.deterministic else True
    cudnn.deterministic = True if args.deterministic else False
    if args.deterministic:
        init_seed(args.local_rank)
        torch.set_printoptions(precision=10)

    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1

    args.gpu = 0
    args.world_size = 1

    if args.distributed:
        args.gpu = args.local_rank
        torch.cuda.set_device(args.gpu)
        torch.distributed.init_process_group(backend=cfg.DIST_BACKEND, init_method=cfg.INIT_METHOD)
        args.world_size = torch.distributed.get_world_size()

    assert torch.backends.cudnn.enabled, "Amp requires cudnn backend to be enabled."


def reduce_tensor(world_size, tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= world_size
    return rt


def is_master_proc():
    """
    Determines if the current process is the master process.
    """
    if torch.distributed.is_initialized():
        return dist.get_rank() % get_world_size() == 0
    else:
        return True


def get_world_size():
    """
    Get the size of the world.
    """
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()
