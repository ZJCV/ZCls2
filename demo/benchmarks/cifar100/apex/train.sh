#!/bin/bash

set -eux

folder="./data"
if [ ! -d ${folder} ]; then
  mkdir ${folder}
fi
cd ${folder}

train_folder="train"
val_folder="val"

if [ ! -d ${train_folder} ]; then
  ln -s /home/zj/data/cifar/train ${train_folder}
fi
if [ ! -d ${val_folder} ]; then
  ln -s /home/zj/data/cifar/test ${val_folder}
fi
cd ..

export PYTHONPATH=.

time1=$(date +%s)
echo "start time: $time1"

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port="15231" \
  main_amp.py --arch resnet18 --pretrained \
  --workers 4 --batch-size 256 --learning-rate 0.1 --momentum 0.9 --weight-decay 1e-4 \
  --epochs 90 --start-epoch 0 \
  --opt-level O1 \
  ${folder}

time2=$(date +%s)
echo "end time: $time2"

train_time=`expr $time2 - $time1`
echo "train time: $train_time"