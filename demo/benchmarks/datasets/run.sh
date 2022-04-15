#!/bin/bash

set -eux

cd ../../../

folder="./data"
if [ ! -d ${folder} ]; then
  mkdir ${folder}
fi

echo "Download CIFAR10 and Extract"
python3 demo/benchmarks/datasets/extract_torchvision_dataset.py "${folder}/cifar10/" --dataset CIFAR10
echo "Download CIFAR100 and Extract"
python3 demo/benchmarks/datasets/extract_torchvision_dataset.py "${folder}/cifar100/" --dataset CIFAR100
echo "Download FashionMNIST and Extract"
python3 demo/benchmarks/datasets/extract_torchvision_dataset.py "${folder}/fashionmnist/" --dataset FashionMNIST