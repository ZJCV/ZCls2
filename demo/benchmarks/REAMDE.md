
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

## Data



## Results

| repos  | arch | dataset  | top1  | top5  | train_time/s  |
|---|---|---|---|---|---|
| apex  | resnet18  | cifar100  | 71.530  | 91.650  | 802  |
| zcls2 | resnet18  | cifar100  | 72.440  | 91.930  | 802  |
|   |   |   |   |   |   |
|   |   |   |   |   |   |
|   |   |   |   |   |   |
|   |   |   |   |   |   |