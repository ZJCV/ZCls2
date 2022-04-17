
# MixDataset Training

Custom a dataset called `MixDataset`, including `CIFAR100/FashionMNIST/VOC2012`. For better learning and use, different models, different loss functions, different optimizers and different learning-rate schedulers are trained based on `MixDataset`.

## Default

Except otherwise noted, all models have been trained on 4x 3090 GPUs with the following parameters:

| Parameter                | value  |
| ------------------------ | ------ |
| `batch_size`             | `256`  |
| `epochs`                 | `90`   |
| `criterion`              | `CrossEntropyLoss`  |
| `optimizer`              | `sgd`  |
| `lr`                     | `0.4`  |
| `momentum`               | `0.9`  |
| `weight-decay`           | `1e-4` |
| `warmup`                 | `5`    |
| `lr-step-size`           | `30/60/80` |
| `lr-gamma`               | `0.1`  |

## ResNet

