#!/bin/bash

set -eux

folder="./data"
dataset_name="cifar10"
arch=mobilenet_v2
num_classes=10

train_sh="./demo/benchmarks/cifar10/apex/main_amp.py"

project_root="/home/zj/repos/ZCls2"

export PYTHONPATH="${project_root}"
echo $PYTHONPATH

cd ${project_root}

time1=$(date +%s)
echo "start time: $time1"

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port="15231" \
  ${train_sh} --arch ${arch} --pretrained --num-classes ${num_classes} \
  --workers 4 --batch-size 256 --learning-rate 0.01 --momentum 0.9 --weight-decay 4e-5 \
  --epochs 90 --start-epoch 0 \
  --opt-level O1 \
  "${folder}/${dataset_name}"

time2=$(date +%s)
echo "end time: $time2"

# shellcheck disable=SC2006
# shellcheck disable=SC2003
train_time=$(expr "$time2" - "$time1")
echo "train time: $train_time"
