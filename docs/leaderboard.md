
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
|  [efficientnet_b0_cccf_600_b64_e90_g4](../configs/cccf/efficientnet_b0_cccf_600_b64_e90_g4.yaml)   |  efficientnet_b0  | 84.596 / 96.751 | CrossEntropyLoss |    SGD    |  MultiStepLR |   90  |   True  |
|  [efficientnet_b2_cccf_260_b256_e90_g4](../configs/cccf/efficientnet_b2_cccf_260_b256_e90_g4.yaml)   |  efficientnet_b2  | 84.472 / 96.725 | CrossEntropyLoss |    SGD    |  MultiStepLR |   90  |   True  |
|  [efficientnet_b1_cccf_224_b256_e90_g4_v2](../configs/cccf/efficientnet_b1_cccf_224_b256_e90_g4_v2.yaml)   |  efficientnet_b1  | 83.749 / 96.524 | LabelSmoothing |    SGD    |  MultiStepLR |   90  |   True  |
|  [mbv3_large_cccf_224_b256_e90_g4_mixup](../../configs/cccf/mbv3_large_cccf_224_b256_e90_g4_mixup.yaml)   | mbv3_large | 82.735 / 96.131 | CrossEntropyLoss |    SGD    |  MultiStepLR |   90  |   True  |
|  [ghostnet_100_cccf_600_b128_e90_g4](../configs/cccf/ghostnet_100_cccf_600_b128_e90_g4.yaml)   | ghostnet_100 | 82.291 / 96.024 | CrossEntropyLoss |    SGD    |  MultiStepLR |   90  |   True  |
|  [r50_cccf_224_b256_e90_g4_calr](../configs/cccf/r50_cccf_224_b256_e90_g4_calr.yaml)   |  resnet50  | 80.101 / 95.979 | CrossEntropyLoss |    SGD    |  CosineAnnealingLR |   90  |   True  |
|  [mbv3_small_cccf_224_b256_e90_g4](../configs/cccf/mbv3_small_cccf_224_b256_e90_g4.yaml)   | mbv3_small | 79.458 / 94.963 | CrossEntropyLoss |    SGD    |  MultiStepLR |   90  |   True  |
|  [ghostnet_130_cccf_224_b256_e90_g4](../configs/cccf/ghostnet_130_cccf_224_b256_e90_g4.yaml)   |  ghostnet_130  | 72.151 / 91.706 | CrossEntropyLoss |    SGD    |  MultiStepLR |   90  |   False  |
|  [efficientnet_lite0_cccf_224_b256_e90_g4](../../configs/cccf/efficientnet_lite0_cccf_224_b256_e90_g4.yaml)   |  efficientnet_lite0  | 69.423 / 90.026 | CrossEntropyLoss |    SGD    |  MultiStepLR |   90  |   False  |
|  [efficientnet_lite2_cccf_260_b256_e90_g4](../../configs/cccf/efficientnet_lite2_cccf_260_b256_e90_g4.yaml)   |  efficientnet_lite2  | 69.294 / 90.194 | CrossEntropyLoss |    SGD    |  MultiStepLR |   90  |   False  |
|  [efficientnet_lite3_cccf_280_b128_e90_g4](../../configs/cccf/efficientnet_lite3_cccf_280_b128_e90_g4.yaml)   |  efficientnet_lite3  | 69.238 / 89.953 | CrossEntropyLoss |    SGD    |  MultiStepLR |   90  |   False  |
|  [efficientnet_lite1_cccf_224_b256_e90_g4](../../configs/cccf/efficientnet_lite1_cccf_224_b256_e90_g4.yaml)   |  efficientnet_lite1  | 68.726 / 89.949 | CrossEntropyLoss |    SGD    |  MultiStepLR |   90  |   False  |
|  [efficientnet_lite4_cccf_300_b128_e90_g4](../../configs/cccf/efficientnet_lite4_cccf_300_b128_e90_g4.yaml)   |  efficientnet_lite4  | 66.856 / 88.343 | CrossEntropyLoss |    SGD    |  MultiStepLR |   90  |   False  |
