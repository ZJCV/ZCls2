
# MobileNet

| cfg |    model   |   top1/top5   |       loss       | optimizer | lr-scheduler | epoch | pretrained |
|:---:|:----------:|:-------------:|:----------------:|:---------:|:------------:|:-----:|:-----:|
|  [mbv3_large_cccf_224_b256_e90_g4_mixup](../../configs/cccf/mbv3_large_cccf_224_b256_e90_g4_mixup.yaml)   | mbv3_large | 82.735 / 96.131 | CrossEntropyLoss |    SGD    |  MultiStepLR |   90  |   True  |
|  [mbv3_large_cccf_224_b256_e90_g4_ra](../../configs/cccf/mbv3_large_cccf_224_b256_e90_g4_ra.yaml)   | mbv3_large | 81.772 / 95.615 | CrossEntropyLoss |    SGD    |  MultiStepLR |   90  |   True  |
|  [mbv3_large_cccf_224_b256_e90_g4](../../configs/cccf/mbv3_large_cccf_224_b256_e90_g4.yaml)   | mbv3_large | 81.117 / 95.437 | CrossEntropyLoss |    SGD    |  MultiStepLR |   90  |   True  |
|  [mbv3_large_cccf_224_b256_e90_g4_ra](../../configs/cccf/mbv3_large_cccf_224_b256_e90_g4_ra.yaml)   | mbv3_large | 79.689 / 94.979 | CrossEntropyLoss |    SGD    |  MultiStepLR |   90  |   True  |
|  [mbv3_large_cccf_224_b256_e90_g4_mixup_v2](../../configs/cccf/mbv3_large_cccf_224_b256_e90_g4_mixup_v2.yaml)   | mbv3_large | 79.598 / 94.864 | CrossEntropyLoss |    SGD    |  MultiStepLR |   90  |   True  |
|  [mbv3_large_cccf_224_b256_e90_g4_no_bias-norm](../../configs/cccf/mbv3_large_cccf_224_b256_e90_g4_no_bias-norm.yaml)   | mbv3_large | 79.56 / 94.90 | CrossEntropyLoss |    SGD    |  MultiStepLR |   90  |   True  |
|  [mbv3_small_cccf_224_b256_e90_g4](../../configs/cccf/mbv3_small_cccf_224_b256_e90_g4.yaml)   | mbv3_small | 79.458 / 94.963 | CrossEntropyLoss |    SGD    |  MultiStepLR |   90  |   True  |
|  [mbv3_large_cccf_224_b256_e90_g4_calr](../../configs/cccf/mbv3_large_cccf_224_b256_e90_g4_calr.yaml)   | mbv3_large | 79.254 / 94.542 | CrossEntropyLoss |    SGD    |  CosineAnnealingLR |   90  |   True  |
|  [mbv3_large_cccf_300_b256_e90_g4](../../configs/cccf/mbv3_large_cccf_300_b256_e90_g4.yaml)   | mbv3_large | 77.087 / 93.724 | CrossEntropyLoss |    SGD    |  MultiStepLR |   90  |   True  |
|  [mbv3_small_cccf_224_b256_e90_g4_mixup](../../configs/cccf/mbv3_small_cccf_224_b256_e90_g4_mixup.yaml)   | mbv3_small | 74.808 / 92.866 | CrossEntropyLoss |    SGD    |  MultiStepLR |   90  |   True  |
|  [mbv3_small_cccf_224_b256_e90_g4_mixup_v2](../../configs/cccf/mbv3_small_cccf_224_b256_e90_g4_mixup_v2.yaml)   | mbv3_small | 74.792 / 92.765 | CrossEntropyLoss |    SGD    |  MultiStepLR |   90  |   True  |
|  [mbv3_small_cccf_224_b256_e90_g4_calr_rmsprop](../../configs/cccf/mbv3_small_cccf_224_b256_e90_g4_calr_rmsprop.yaml)   |  mbv3_small  | 69.081 / 89.949 | CrossEntropyLoss |    RMSProp    |  CosineAnnealingLR |   90  |   True  |
|  [mbv3_large_cccf_224_b256_e90_g4_calr_rmsprop](../../configs/cccf/mbv3_large_cccf_224_b256_e90_g4_calr_rmsprop.yaml)   |  mbv3_large  | 61.297 / 85.187 | CrossEntropyLoss |    RMSProp    |  CosineAnnealingLR |   90  |   True  |
