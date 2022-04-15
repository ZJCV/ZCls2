#!/bin/bash

set -eux

cd ../../../

folder="./data"
if [ ! -d ${folder} ]; then
  mkdir ${folder}
fi

python3 demo/benchmarks/datasets/extract_torchvision_dataset.py "${folder}/cifar10/" --dataset CIFAR10
python3 demo/benchmarks/datasets/extract_torchvision_dataset.py "${folder}/cifar100/" --dataset CIFAR100
python3 demo/benchmarks/datasets/extract_torchvision_dataset.py "${folder}/fashionmnist/" --dataset FashionMNIST