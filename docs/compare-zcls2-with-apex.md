
# Benchmarks

>Compare zcls2 and apex training

## Requirements

```markdown
* Model: Torchvision Pretrained ResNet18
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
```

## Prepare data

Go to the `./demo/benchmarks/datasets` folder, run

```shell
bash run.sh
```

This script will download torchvision data and extract to `./data/` folder.

```
ZCls2/data$ tree -L 2
.
├── cifar10
│   ├── cifar-10-batches-py
│   ├── cifar-10-python.tar.gz
│   ├── train
│   └── val
├── cifar100
│   ├── cifar-100-python
│   ├── cifar-100-python.tar.gz
│   ├── train
│   └── val
├── cifar-10-python.tar.gz
└── fashionmnist
    ├── FashionMNIST
    │   ├── processed
    │   └── raw
    ├── train
    │   ├── Ankle boot
    │   ├── Bag
    │   ├── Coat
    │   ├── Dress
    │   ├── Pullover
    │   ├── Sandal
    │   ├── Shirt
    │   ├── Sneaker
    │   ├── Trouser
    │   └── T-shirt
    └── val
        ├── Ankle boot
        ├── Bag
        ├── Coat
        ├── Dress
        ├── Pullover
        ├── Sandal
        ├── Shirt
        ├── Sneaker
        ├── Trouser
        └── T-shirt
```

## Results

| repos  | arch | dataset  | top1  | top5  | train_time  |
|---|---|---|---|---|---|
| apex  | resnet18  | cifar10  | 84.940  |  99.360 | 797  |
| zcls2 | resnet18  | cifar10  |  87.490 | 99.540  | 797  |
| apex  | resnet18  | cifar100  | 72.760  | 92.350  | 807  |
| zcls2 | resnet18  | cifar100  | 72.760   | 92.070  | 797  |
| apex  | resnet18  | fashionmnist  | 88.830  | 99.880 | 902  |
| zcls2 | resnet18  | fashionmnist  | 94.250   | 99.970  | 907  |

I don't set `cudnn.deterministic = True` and `cudnn.benchmark = False`, so each time the `best_prec@1/best_prec@5` is different, may be big diff. 

For example, FashionMNIST in ZCls2:

1. `94.250 / 99.970`
2. `88.150 / 99.850`
3. `94.240 / 99.960`

CIFAR10 in ZCls2:

1. `87.490 / 99.540`
2. `79.170 / 98.610`
3. `82.780 / 99.260`

CIFAR10 in Apex:

1. `84.940 / 99.360`
2. `83.120 / 99.250`
3. `85.709 / 99.350`