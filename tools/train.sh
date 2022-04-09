#!/bin/bash

export PYTHONPATH=.

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port="15231" \
  tools/train.py -cfg configs/r18_cifar100_224_b246_e90_g4.yaml \
  --opt-level O1
