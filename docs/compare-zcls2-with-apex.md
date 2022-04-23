
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
    * Optimizer: SGD (initial lr 0.01)
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

| repos  | arch | dataset  | top1  | top5  |
|---|---|---|---|---|
| apex  | resnet18  | cifar10  | 92.910 | 99.800 |
| zcls2 | resnet18  | cifar10  | 92.410 | 99.770 |
| apex  | resnet18  | cifar100 | 73.250 | 92.890 |
| zcls2 | resnet18  | cifar100 | 73.190 | 93.080 |
| apex  | resnet18  | fashionmnist  | 94.230 | 99.950 |
| zcls2 | resnet18  | fashionmnist  | 93.920 | 99.930 |

I don't set `cudnn.deterministic = True` and `cudnn.benchmark = False`, so each time the `best_prec@1/best_prec@5` is different, may be big diff. For example, 

### CIFAR10

1. Apex
   1. `92.910 | 99.800`
2. ZCls2
   1. `92.410 | 99.770`

### CIFAR100

1. Apex
   1. `73.250 | 92.890`
2. ZCls2
   1. `73.190 | 93.080`

### FashionMNIST

1. Apex
   1. `94.230 | 99.950`
2. ZCls2
   1. `93.920 | 99.930`
