
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
|  [efficientnet_b7_cccf_600_b8_e90_g4](../configs/cccf/efficientnet_b7_cccf_600_b8_e90_g4.yaml)   |  efficientnet_b7  | 88.118 / 97.676 | CrossEntropyLoss |    SGD    |  MultiStepLR |   90  |   True  |
|  [efficientnet_b6_cccf_528_b16_e90_g4](../configs/cccf/efficientnet_b6_cccf_528_b16_e90_g4.yaml)   |  efficientnet_b6  | 87.992 / 97.611 | CrossEntropyLoss |    SGD    |  MultiStepLR |   90  |   True  |
|  [efficientnet_b4_cccf_380_b64_e90_g4](../configs/cccf/efficientnet_b4_cccf_380_b64_e90_g4.yaml)   |  efficientnet_b4  | 87.693 / 97.716 | CrossEntropyLoss |    SGD    |  MultiStepLR |   90  |   True  |
|  [efficientnet_b5_cccf_456_b32_e90_g4](../configs/cccf/efficientnet_b5_cccf_456_b32_e90_g4.yaml)   |  efficientnet_b5  | 87.557 / 97.606 | CrossEntropyLoss |    SGD    |  MultiStepLR |   90  |   True  |
|  [efficientnet_b3_cccf_300_b128_e90_g4](../configs/cccf/efficientnet_b3_cccf_300_b128_e90_g4.yaml)   |  efficientnet_b3  | 85.704 / 97.043 | CrossEntropyLoss |    SGD    |  MultiStepLR |   90  |   True  |
|  [efficientnet_b4_cccf_224_b256_e90_g4](../configs/cccf/efficientnet_b4_cccf_224_b256_e90_g4.yaml)   |  efficientnet_b4  | 85.587 / 97.134 | CrossEntropyLoss |    SGD    |  MultiStepLR |   90  |   True  |
|  [efficientnet_b6_cccf_224_b256_e90_g4](../configs/cccf/efficientnet_b6_cccf_224_b256_e90_g4.yaml)   |  efficientnet_b6  | 85.133 / 96.788 | CrossEntropyLoss |    SGD    |  MultiStepLR |   90  |   True  |
|  [efficientnet_b7_cccf_224_b256_e90_g4](../configs/cccf/efficientnet_b7_cccf_224_b256_e90_g4.yaml)   |  efficientnet_b7  | 85.122 / 96.791 | CrossEntropyLoss |    SGD    |  MultiStepLR |   90  |   True  |
|  [efficientnet_b5_cccf_224_b256_e90_g4](../configs/cccf/efficientnet_b5_cccf_224_b256_e90_g4.yaml)   |  efficientnet_b5  | 84.848 / 96.713 | CrossEntropyLoss |    SGD    |  MultiStepLR |   90  |   True  |
|  [efficientnet_b0_cccf_600_b64_e90_g4](../configs/cccf/efficientnet_b0_cccf_600_b64_e90_g4.yaml)   |  efficientnet_b0  | 84.596 / 96.751 | CrossEntropyLoss |    SGD    |  MultiStepLR |   90  |   True  |
|  [efficientnet_b2_cccf_260_b256_e90_g4](../configs/cccf/efficientnet_b2_cccf_260_b256_e90_g4.yaml)   |  efficientnet_b2  | 84.472 / 96.725 | CrossEntropyLoss |    SGD    |  MultiStepLR |   90  |   True  |
|  [efficientnet_b3_cccf_224_b256_e90_g4](../configs/cccf/efficientnet_b3_cccf_224_b256_e90_g4.yaml)   |  efficientnet_b3  | 84.392 / 96.678 | CrossEntropyLoss |    SGD    |  MultiStepLR |   90  |   True  |
|  [efficientnet_b1_cccf_224_b256_e90_g4_v2](../configs/cccf/efficientnet_b1_cccf_224_b256_e90_g4_v2.yaml)   |  efficientnet_b1  | 83.749 / 96.524 | LabelSmoothing |    SGD    |  MultiStepLR |   90  |   True  |
|  [efficientnet_b1_cccf_240_b256_e90_g4](../configs/cccf/efficientnet_b1_cccf_240_b256_e90_g4.yaml)   |  efficientnet_b1  | 83.682 / 96.377 | CrossEntropyLoss |    SGD    |  MultiStepLR |   90  |   True  |
|  [efficientnet_b1_cccf_224_b256_e90_g4](../configs/cccf/efficientnet_b1_cccf_224_b256_e90_g4.yaml)   |  efficientnet_b1  | 83.523 / 96.300 | CrossEntropyLoss |    SGD    |  MultiStepLR |   90  |   True  |
|  [efficientnet_b2_cccf_224_b256_e90_g4](../configs/cccf/efficientnet_b2_cccf_224_b256_e90_g4.yaml)   |  efficientnet_b2  | 83.478 / 96.323 | CrossEntropyLoss |    SGD    |  MultiStepLR |   90  |   True  |
|  [efficientnet_b1_cccf_224_b256_e90_g4_mixup](../configs/cccf/efficientnet_b1_cccf_224_b256_e90_g4_mixup.yaml)   |  efficientnet_b1  | 83.326 / 96.363 | CrossEntropyLoss |    SGD    |  MultiStepLR |   90  |   True  |
|  [efficientnet_b0_cccf_300_b256_e90_g4](../configs/cccf/efficientnet_b0_cccf_300_b256_e90_g4.yaml)   |  efficientnet_b0  | 83.195 / 96.374 | CrossEntropyLoss |    SGD    |  MultiStepLR |   90  |   True  |
|  [ghostnet_100_cccf_600_b128_e90_g4](../configs/cccf/ghostnet_100_cccf_600_b128_e90_g4.yaml)   | ghostnet_100 | 82.291 / 96.024 | CrossEntropyLoss |    SGD    |  MultiStepLR |   90  |   True  |
|  [efficientnet_b0_cccf_224_b256_e90_g4](../configs/cccf/efficientnet_b0_cccf_224_b256_e90_g4.yaml)   |  efficientnet_b0  | 82.034 / 96.010 | CrossEntropyLoss |    SGD    |  MultiStepLR |   90  |   True  |
|  [efficientnet_b0_cccf_224_b256_e90_g4_mixup](../configs/cccf/efficientnet_b0_cccf_224_b256_e90_g4_mixup.yaml)   |  efficientnet_b0  | 81.987 / 95.912 | CrossEntropyLoss |    SGD    |  MultiStepLR |   90  |   True  |
|  [efficientnet_b0_cccf_224_b256_e90_g4_calr](../configs/cccf/efficientnet_b0_cccf_224_b256_e90_g4_calr.yaml)   |  efficientnet_b0  | 81.938 / 95.865 | CrossEntropyLoss |    SGD    |  CosineAnnealingLR |   90  |   True  |
|  [efficientnet_b0_cccf_224_b256_e90_g4_v2](../configs/cccf/efficientnet_b0_cccf_224_b256_e90_g4_v2.yaml)   |  efficientnet_b0  | 81.599 / 95.710 | LabelSmoothing |    SGD    |  MultiStepLR |   90  |   True  |
|  [efficientnet_b0_cccf_224_b256_e90_g4_ft](../configs/cccf/efficientnet_b0_cccf_224_b256_e90_g4_ft.yaml)   |  efficientnet_b0  | 81.403 / 95.914 | CrossEntropyLoss |    SGD    |  CosineAnnealingLR |   90  |   True  |
|  [ghostnet_100_cccf_300_b256_e90_g4](../configs/cccf/ghostnet_100_cccf_300_b256_e90_g4.yaml)   | ghostnet_100 | 80.972 / 95.460 | CrossEntropyLoss |    SGD    |  MultiStepLR |   90  |   True  |
|  [r50_cccf_224_b256_e90_g4_calr](../configs/cccf/r50_cccf_224_b256_e90_g4_calr.yaml)   |  resnet50  | 80.101 / 95.979 | CrossEntropyLoss |    SGD    |  CosineAnnealingLR |   90  |   True  |
|  [ghostnet_100_cccf_224_b256_e90_g4](../configs/cccf/ghostnet_100_cccf_224_b256_e90_g4.yaml)   | ghostnet_100 | 79.801 / 95.00 | CrossEntropyLoss |    SGD    |  MultiStepLR |   90  |   True  |
|  [mbv3_large_cccf_224_b256_e90_g4_mixup_v2](../configs/cccf/mbv3_large_cccf_224_b256_e90_g4_mixup_v2.yaml)   | mbv3_large | 79.598 / 94.864 | CrossEntropyLoss |    SGD    |  MultiStepLR |   90  |   True  |
|  [mbv3_large_cccf_224_b256_e90_g4](../configs/cccf/mbv3_large_cccf_224_b256_e90_g4.yaml)   | mbv3_large | 79.56 / 94.90 | CrossEntropyLoss |    SGD    |  MultiStepLR |   90  |   True  |
|  [mbv3_small_cccf_224_b256_e90_g4](../configs/cccf/mbv3_small_cccf_224_b256_e90_g4.yaml)   | mbv3_small | 79.458 / 94.963 | CrossEntropyLoss |    SGD    |  MultiStepLR |   90  |   True  |
|  [mbv3_large_cccf_224_b256_e90_g4_calr](../configs/cccf/mbv3_large_cccf_224_b256_e90_g4_calr.yaml)   | mbv3_large | 79.254 / 94.542 | CrossEntropyLoss |    SGD    |  CosineAnnealingLR |   90  |   True  |
|  [ghostnet_100_cccf_224_b256_e90_g4_ft](../configs/cccf/ghostnet_100_cccf_224_b256_e90_g4_ft.yaml)   | ghostnet_100 | 78.373 / 94.813 | CrossEntropyLoss |    SGD    |  MultiStepLR |   90  |   True  |
|  [r50_cccf_224_b256_e90_g4](../configs/cccf/r50_cccf_224_b256_e90_g4.yaml)   |  resnet50  | 77.11 / 93.93 | CrossEntropyLoss |    SGD    |  MultiStepLR |   90  |   True  |
|  [mbv3_large_cccf_300_b256_e90_g4](../configs/cccf/mbv3_large_cccf_300_b256_e90_g4.yaml)   | mbv3_large | 77.087 / 93.724 | CrossEntropyLoss |    SGD    |  MultiStepLR |   90  |   True  |
|  [mbv3_small_cccf_224_b256_e90_g4_mixup](../configs/cccf/mbv3_small_cccf_224_b256_e90_g4_mixup.yaml)   | mbv3_small | 74.808 / 92.866 | CrossEntropyLoss |    SGD    |  MultiStepLR |   90  |   True  |
|  [mbv3_small_cccf_224_b256_e90_g4_mixup_v2](../configs/cccf/mbv3_small_cccf_224_b256_e90_g4_mixup_v2.yaml)   | mbv3_small | 74.792 / 92.765 | CrossEntropyLoss |    SGD    |  MultiStepLR |   90  |   True  |
|  [mbv3_large_cccf_224_b256_e90_g4_mixup](../configs/cccf/mbv3_large_cccf_224_b256_e90_g4_mixup.yaml)   | mbv3_large | 73.462 / 92.022 | CrossEntropyLoss |    SGD    |  MultiStepLR |   90  |   True  |
|  [ghostnet_130_cccf_224_b256_e90_g4](../configs/cccf/ghostnet_130_cccf_224_b256_e90_g4.yaml)   |  ghostnet_130  | 72.151 / 91.706 | CrossEntropyLoss |    SGD    |  MultiStepLR |   90  |   False  |
|  [ghostnet_130_cccf_224_b256_e90_g4_v2](../configs/cccf/ghostnet_130_cccf_224_b256_e90_g4_v2.yaml)   |  ghostnet_130  | 70.089 / 90.507 | CrossEntropyLoss |    SGD    |  MultiStepLR |   90  |   False  |
|  [mbv3_small_cccf_224_b256_e90_g4_calr_rmsprop](../configs/cccf/mbv3_small_cccf_224_b256_e90_g4_calr_rmsprop.yaml)   |  mbv3_small  | 69.081 / 89.949 | CrossEntropyLoss |    RMSProp    |  CosineAnnealingLR |   90  |   True  |
|  [efficientnet_lite0_cccf_300_b256_e90_g4](../configs/cccf/efficientnet_lite0_cccf_300_b256_e90_g4.yaml)   |  efficientnet_lite0  | 69.196 / 89.928 | CrossEntropyLoss |    SGD    |  MultiStepLR |   90  |   False  |
|  [efficientnet_lite0_cccf_224_b256_e90_g4](../configs/cccf/efficientnet_lite0_cccf_224_b256_e90_g4.yaml)   |  efficientnet_lite0  | 68.230 / 89.423 | CrossEntropyLoss |    SGD    |  MultiStepLR |   90  |   False  |
|  [efficientnet_lite0_cccf_224_b256_e90_g4_mixup](../configs/cccf/efficientnet_lite0_cccf_224_b256_e90_g4_mixup.yaml)   |  efficientnet_lite0  | 67.749 / 89.140 | CrossEntropyLoss |    SGD    |  MultiStepLR |   90  |   False  |
|  [efficientnet_lite2_cccf_260_b256_e90_g4](../configs/cccf/efficientnet_lite2_cccf_260_b256_e90_g4.yaml)   |  efficientnet_lite2  | 66.999 / 88.597 | CrossEntropyLoss |    SGD    |  MultiStepLR |   90  |   False  |
|  [efficientnet_lite1_cccf_240_b256_e90_g4](../configs/cccf/efficientnet_lite1_cccf_240_b256_e90_g4.yaml)   |  efficientnet_lite1  | 66.755 / 88.228 | CrossEntropyLoss |    SGD    |  MultiStepLR |   90  |   False  |
|  [efficientnet_lite1_cccf_224_b256_e90_g4](../configs/cccf/efficientnet_lite1_cccf_224_b256_e90_g4.yaml)   |  efficientnet_lite1  | 66.272 / 88.207 | CrossEntropyLoss |    SGD    |  MultiStepLR |   90  |   False  |
|  [efficientnet_lite0_cccf_224_b256_e90_g4_v2](../configs/cccf/efficientnet_lite0_cccf_224_b256_e90_g4_v2.yaml)   |  efficientnet_lite0  | 66.400 / 88.326 | CrossEntropyLoss |    SGD    |  MultiStepLR |   90  |   False  |
|  [efficientnet_lite3_cccf_280_b256_e90_g4](../configs/cccf/efficientnet_lite3_cccf_280_b256_e90_g4.yaml)   |  efficientnet_lite3  | 66.108 / 87.999 | CrossEntropyLoss |    SGD    |  MultiStepLR |   90  |   False  |
|  [efficientnet_lite2_cccf_224_b256_e90_g4](../configs/cccf/efficientnet_lite2_cccf_224_b256_e90_g4.yaml)   |  efficientnet_lite2  | 65.676 / 87.948 | CrossEntropyLoss |    SGD    |  MultiStepLR |   90  |   False  |
|  [efficientnet_lite3_cccf_224_b256_e90_g4](../configs/cccf/efficientnet_lite3_cccf_224_b256_e90_g4.yaml)   |  efficientnet_lite3  | 65.182 / 87.679 | CrossEntropyLoss |    SGD    |  MultiStepLR |   90  |   False  |
|  [efficientnet_lite1_cccf_224_b256_e90_g4_v2](../configs/cccf/efficientnet_lite1_cccf_224_b256_e90_g4_v2.yaml)   |  efficientnet_lite1  | 64.469 / 87.092 | CrossEntropyLoss |    SGD    |  MultiStepLR |   90  |   False  |
|  [efficientnet_lite4_cccf_300_b256_e90_g4](../configs/cccf/efficientnet_lite4_cccf_300_b256_e90_g4.yaml)   |  efficientnet_lite4  | 62.422 / 85.358 | CrossEntropyLoss |    SGD    |  MultiStepLR |   90  |   False  |
|  [efficientnet_lite4_cccf_224_b256_e90_g4](../configs/cccf/efficientnet_lite4_cccf_224_b256_e90_g4.yaml)   |  efficientnet_lite4  | 62.062 / 85.346 | CrossEntropyLoss |    SGD    |  MultiStepLR |   90  |   False  |
|  [efficientnet_lite4_cccf_300_b256_e90_g4_adam](../configs/cccf/efficientnet_lite4_cccf_300_b256_e90_g4_adam.yaml)   |  efficientnet_lite4  | 62.003 / 85.409 | CrossEntropyLoss |    SGD    |  MultiStepLR |   90  |   False  |
|  [mbv3_large_cccf_224_b256_e90_g4_calr_rmsprop](../configs/cccf/mbv3_large_cccf_224_b256_e90_g4_calr_rmsprop.yaml)   |  mbv3_large  | 61.297 / 85.187 | CrossEntropyLoss |    RMSProp    |  CosineAnnealingLR |   90  |   True  |
