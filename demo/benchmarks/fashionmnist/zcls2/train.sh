#!/bin/bash

set -eux

cd ../../../../

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

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port="21231" \
  tools/train.py -cfg demo/benchmarks/cifar10/zcls2/r18_cifar10_224_b256_e90_g4.yaml \
  --opt-level O1

time2=$(date +%s)
echo "end time: $time2"

train_time=`expr $time2 - $time1`
echo "train time: $train_time"