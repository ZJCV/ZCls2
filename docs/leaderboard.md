
# LeaderBoard (Based on CCCF)

## About CCCF

    CCCF is a custom mixed classification dataset, including

    1. CIFAR100: https://paperswithcode.com/dataset/cifar-100
    2. CUB-200-2011: https://paperswithcode.com/dataset/cub-200-2011
    3. Caltech-101: https://paperswithcode.com/dataset/caltech-101
    4. Food-101: https://paperswithcode.com/dataset/food-101

    The classes num = 100 + 200 + 101 + 101 = 502

### SCORES

| cfg |    model   |   top1/top5   |       loss       | optimizer | lr-scheduler | epoch | pretrained |
|:---:|:----------:|:-------------:|:----------------:|:---------:|:------------:|:-----:|:-----:|
|  [efficientnet_b4_cccf_224_b256_e90_g4](../configs/cccf/efficientnet_b4_cccf_224_b256_e90_g4.yaml)   |  efficientnet_b4  | 85.587 / 97.134 | CrossEntropyLoss |    SGD    |  MultiStepLR |   90  |   True  |
|  [efficientnet_b3_cccf_224_b256_e90_g4](../configs/cccf/efficientnet_b3_cccf_224_b256_e90_g4.yaml)   |  efficientnet_b3  | 84.392 / 96.678 | CrossEntropyLoss |    SGD    |  MultiStepLR |   90  |   True  |
|  [efficientnet_b1_cccf_224_b256_e90_g4](../configs/cccf/efficientnet_b1_cccf_224_b256_e90_g4.yaml)   |  efficientnet_b1  | 83.523 / 96.300 | CrossEntropyLoss |    SGD    |  MultiStepLR |   90  |   True  |
|  [efficientnet_b2_cccf_224_b256_e90_g4](../configs/cccf/efficientnet_b2_cccf_224_b256_e90_g4.yaml)   |  efficientnet_b2  | 83.478 / 96.323 | CrossEntropyLoss |    SGD    |  MultiStepLR |   90  |   True  |
|  [efficientnet_b0_cccf_224_b256_e90_g4](../configs/cccf/efficientnet_b0_cccf_224_b256_e90_g4.yaml)   |  efficientnet_b0  | 82.034 / 96.010 | CrossEntropyLoss |    SGD    |  MultiStepLR |   90  |   True  |
|  [efficientnet_b0_cccf_224_b256_e90_g4_calr](../configs/cccf/efficientnet_b0_cccf_224_b256_e90_g4_calr.yaml)   |  efficientnet_b0  | 81.938 / 95.865 | CrossEntropyLoss |    SGD    |  CosineAnnealingLR |   90  |   True  |
|  [r50_cccf_224_b256_e90_g4_calr](../configs/cccf/r50_cccf_224_b256_e90_g4_calr.yaml)   |  resnet50  | 80.101 / 95.979 | CrossEntropyLoss |    SGD    |  CosineAnnealingLR |   90  |   True  |
|  [ghostnet_100_cccf_224_b256_e90_g4](../configs/cccf/ghostnet_100_cccf_224_b256_e90_g4.yaml)   | ghostnet_100 | 79.801 / 95.00 | CrossEntropyLoss |    SGD    |  MultiStepLR |   90  |   True  |
|  [mbv3_large_cccf_224_b256_e90_g4](../configs/cccf/mbv3_large_cccf_224_b256_e90_g4.yaml)   | mbv3_large | 79.56 / 94.90 | CrossEntropyLoss |    SGD    |  MultiStepLR |   90  |   True  |
|  [mbv3_small_cccf_224_b256_e90_g4](../configs/cccf/mbv3_small_cccf_224_b256_e90_g4.yaml)   | mbv3_small | 79.458 / 94.963 | CrossEntropyLoss |    SGD    |  MultiStepLR |   90  |   True  |
|  [mbv3_large_cccf_224_b256_e90_g4_calr](../configs/cccf/mbv3_large_cccf_224_b256_e90_g4_calr.yaml)   | mbv3_large | 79.254 / 94.542 | CrossEntropyLoss |    SGD    |  CosineAnnealingLR |   90  |   True  |
|  [r50_cccf_224_b256_e90_g4](../configs/cccf/r50_cccf_224_b256_e90_g4.yaml)   |  resnet50  | 77.11 / 93.93 | CrossEntropyLoss |    SGD    |  MultiStepLR |   90  |   True  |
|  [ghostnet_130_cccf_224_b256_e90_g4](../configs/cccf/ghostnet_130_cccf_224_b256_e90_g4.yaml)   |  ghostnet_130  | 72.151 / 91.706 | CrossEntropyLoss |    SGD    |  MultiStepLR |   90  |   False  |
|  [mbv3_small_cccf_224_b256_e90_g4_calr_rmsprop](../configs/cccf/mbv3_small_cccf_224_b256_e90_g4_calr_rmsprop.yaml)   |  mbv3_small  | 69.081 / 89.949 | CrossEntropyLoss |    RMSProp    |  CosineAnnealingLR |   90  |   True  |
|  [efficientnet_lite0_cccf_224_b256_e90_g4](../configs/cccf/efficientnet_lite0_cccf_224_b256_e90_g4.yaml)   |  efficientnet_lite0  | 68.230 / 89.423 | CrossEntropyLoss |    SGD    |  MultiStepLR |   90  |   False  |
|  [efficientnet_lite0_cccf_224_b256_e90_g4_mixup](../configs/cccf/efficientnet_lite0_cccf_224_b256_e90_g4_mixup.yaml)   |  efficientnet_lite0  | 67.749 / 89.140 | CrossEntropyLoss |    SGD    |  MultiStepLR |   90  |   False  |
|  [efficientnet_lite1_cccf_224_b256_e90_g4](../configs/cccf/efficientnet_lite1_cccf_224_b256_e90_g4.yaml)   |  efficientnet_lite1  | 66.272 / 88.207 | CrossEntropyLoss |    SGD    |  MultiStepLR |   90  |   False  |
|  [efficientnet_lite2_cccf_224_b256_e90_g4](../configs/cccf/efficientnet_lite2_cccf_224_b256_e90_g4.yaml)   |  efficientnet_lite2  | 65.676 / 87.948 | CrossEntropyLoss |    SGD    |  MultiStepLR |   90  |   False  |
|  [efficientnet_lite3_cccf_224_b256_e90_g4](../configs/cccf/efficientnet_lite3_cccf_224_b256_e90_g4.yaml)   |  efficientnet_lite3  | 65.182 / 87.679 | CrossEntropyLoss |    SGD    |  MultiStepLR |   90  |   False  |
|  [efficientnet_lite4_cccf_224_b256_e90_g4](../configs/cccf/efficientnet_lite4_cccf_224_b256_e90_g4.yaml)   |  efficientnet_lite3  | 62.062 / 85.346 | CrossEntropyLoss |    SGD    |  MultiStepLR |   90  |   False  |
|  [mbv3_large_cccf_224_b256_e90_g4_calr_rmsprop](../configs/cccf/mbv3_large_cccf_224_b256_e90_g4_calr_rmsprop.yaml)   |  mbv3_large  | 61.297 / 85.187 | CrossEntropyLoss |    RMSProp    |  CosineAnnealingLR |   90  |   True  |
