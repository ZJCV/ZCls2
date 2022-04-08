#!/bin/bash

export PYTHONPATH=.

#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port="16231" \
#        tools/train.py -cfg configs/r18_cifar100_224_b246_e90_g4.yaml -a resnet18 --pretrained \
#        --b 256 --workers 4 --dataset GeneralDatasetV2 --opt-level O1 \
#        /home/zj/data/cifar/zcls_cifar

#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port="16231" \
#        tools/train.py -cfg configs/r18_cifar100_224_b246_e90_g4.yaml -a resnet18 --pretrained \
#        --b 256 --workers 4 --opt-level O1

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port="16231" \
        tools/train.py -cfg configs/r18_cifar100_224_b246_e90_g4.yaml -a resnet18 --pretrained \
        --opt-level O1