
# LeaderBoard (Based on CCCF)

| cfg |    model   |   top1/top5   |       loss       | optimizer | lr-scheduler | epoch |
|:---:|:----------:|:-------------:|:----------------:|:---------:|:------------:|:-----:|
|  [mbv3_small_cccf_224_b256_e90_g4](../configs/cccf/mbv3_small_cccf_224_b256_e90_g4.yaml)   | mbv3_small | 79.458 / 94.963 | CrossEntropyLoss |    SGD    |  MultiStepLR |   90  |
|  [mbv3_large_cccf_224_b256_e90_g4](../configs/cccf/mbv3_large_cccf_224_b256_e90_g4.yaml)   | mbv3_large | 79.56 / 94.90 | CrossEntropyLoss |    SGD    |  MultiStepLR |   90  |
|  [mbv3_large_cccf_224_b256_e90_g4_calr](../configs/cccf/mbv3_large_cccf_224_b256_e90_g4_calr.yaml)   | mbv3_large | 79.254 / 94.542 | CrossEntropyLoss |    SGD    |  CosineAnnealingLR |   90  |
|  [r50_cccf_224_b256_e90_g4](../configs/cccf/r50_cccf_224_b256_e90_g4.yaml)   |  resnet50  | 77.11 / 93.93 | CrossEntropyLoss |    SGD    |  MultiStepLR |   90  |
|     |            |               |                  |           |              |       |
