
# Benchmarks

>Compare zcls2 and apex training

## Requirements

* Model: Torchvision Pretrained ResNet18/ResNet50
* Dataset: CIFAR10/CIFAR100/FashionMNIST
* Train:
  * Epoch: 90
  * Num gpus: 4
  * Batch size: 256 / one gpu
  * Loss: CrossEntropyLoss
  * Optimizer: SGD (initial lr 0.1)
  * Lr_scheduler: Warmup (5) + MultiStepLR(30/60/80)
* apex: commit `727a6452c9b781930acee5e24e09efe9360b4890`
* zcls2: commit `33de745bc6ab4fdedb07754c075cec13a7ce16be`

## Prepare data

Go to the `/path/to/demo/benchmarks/datasets` folder, run

```shell
$ bash run.sh
```

This script will download torchvision data and extract to `./data/` folder.

## Results

| repos  | arch | dataset  | top1  | top5  | train_time  |
|---|---|---|---|---|---|
| apex  | resnet18  | cifar10  | 84.940  |  99.360 | 797  |
| zcls2 | resnet18  | cifar10  |  87.490 | 99.540  | 797  |
| apex  | resnet18  | cifar100  | 72.760  | 92.350  | 807  |
| zcls2 | resnet18  | cifar100  | 72.760   | 92.070  | 797  |
| apex  | resnet18  | fashionmnist  | 88.830  | 99.880 | 902  |
| zcls2 | resnet18  | fashionmnist  | 94.250   | 99.970  | 907  |