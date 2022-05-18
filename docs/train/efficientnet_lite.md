
# EfficientNet_Lite

| cfg |    model   |   top1/top5   |       loss       | optimizer | lr-scheduler | epoch | pretrained |
|:---:|:----------:|:-------------:|:----------------:|:---------:|:------------:|:-----:|:-----:|
|  [efficientnet_lite0_cccf_224_b256_e90_g4](../../configs/cccf/efficientnet_lite0_cccf_224_b256_e90_g4.yaml)   |  efficientnet_lite0  | 69.423 / 90.026 | CrossEntropyLoss |    SGD    |  MultiStepLR |   90  |   False  |
|  [efficientnet_lite2_cccf_260_b256_e90_g4](../../configs/cccf/efficientnet_lite2_cccf_260_b256_e90_g4.yaml)   |  efficientnet_lite2  | 69.294 / 90.194 | CrossEntropyLoss |    SGD    |  MultiStepLR |   90  |   False  |
|  [efficientnet_lite3_cccf_280_b128_e90_g4](../../configs/cccf/efficientnet_lite3_cccf_280_b128_e90_g4.yaml)   |  efficientnet_lite3  | 69.238 / 89.953 | CrossEntropyLoss |    SGD    |  MultiStepLR |   90  |   False  |
|  [efficientnet_lite0_cccf_300_b256_e90_g4](../../configs/cccf/efficientnet_lite0_cccf_300_b256_e90_g4.yaml)   |  efficientnet_lite0  | 69.196 / 89.928 | CrossEntropyLoss |    SGD    |  MultiStepLR |   90  |   False  |
|  [efficientnet_lite0_cccf_224_b256_e90_g4_no_bias-norm](../../configs/cccf/efficientnet_lite0_cccf_224_b256_e90_g4_no_bias-norm.yaml)   |  efficientnet_lite0  | 68.822 / 89.970 | CrossEntropyLoss |    SGD    |  MultiStepLR |   90  |   False  |
|  [efficientnet_lite1_cccf_224_b256_e90_g4](../../configs/cccf/efficientnet_lite1_cccf_224_b256_e90_g4.yaml)   |  efficientnet_lite1  | 68.726 / 89.949 | CrossEntropyLoss |    SGD    |  MultiStepLR |   90  |   False  |
|  [efficientnet_lite0_cccf_224_b256_e90_g4_ls](../../configs/cccf/efficientnet_lite0_cccf_224_b256_e90_g4_ls.yaml)   |  efficientnet_lite0  | 68.495 / 89.722 | LabelSmoothingLoss |    SGD    |  MultiStepLR |   90  |   False  |
|  [efficientnet_lite0_cccf_224_b256_e90_g4_ls_no_bias-norm](../../configs/cccf/efficientnet_lite0_cccf_224_b256_e90_g4_ls_no_bias-norm.yaml)   |  efficientnet_lite0  | 68.357 / 89.345 | LabelSmoothingLoss |    SGD    |  MultiStepLR |   90  |   False  |
|  [efficientnet_lite1_cccf_240_b256_e90_g4](../../configs/cccf/efficientnet_lite1_cccf_240_b256_e90_g4.yaml)   |  efficientnet_lite1  | 68.006 / 89.107 | CrossEntropyLoss |    SGD    |  MultiStepLR |   90  |   False  |
|  [efficientnet_lite0_cccf_224_b256_e90_g4_mixup](../../configs/cccf/efficientnet_lite0_cccf_224_b256_e90_g4_mixup.yaml)   |  efficientnet_lite0  | 67.835 / 89.338 | CrossEntropyLoss |    SGD    |  MultiStepLR |   90  |   False  |
|  [efficientnet_lite3_cccf_224_b256_e90_g4](../../configs/cccf/efficientnet_lite3_cccf_224_b256_e90_g4.yaml)   |  efficientnet_lite3  | 67.669 / 89.324 | CrossEntropyLoss |    SGD    |  MultiStepLR |   90  |   False  |
|  [efficientnet_lite1_cccf_224_b256_e90_g4_ls](../../configs/cccf/efficientnet_lite1_cccf_224_b256_e90_g4_ls.yaml)   |  efficientnet_lite1  | 67.609 / 89.226 | CrossEntropyLoss |    SGD    |  MultiStepLR |   90  |   False  |
|  [efficientnet_lite2_cccf_224_b256_e90_g4](../../configs/cccf/efficientnet_lite2_cccf_224_b256_e90_g4.yaml)   |  efficientnet_lite2  | 67.055 / 88.887 | CrossEntropyLoss |    SGD    |  MultiStepLR |   90  |   False  |
|  [efficientnet_lite2_cccf_260_b256_e90_g4_no_bias-norm](../../configs/cccf/efficientnet_lite2_cccf_260_b256_e90_g4_no_bias-norm.yaml)   |  efficientnet_lite2  | 66.999 / 88.597 | CrossEntropyLoss |    SGD    |  MultiStepLR |   90  |   False  |
|  [efficientnet_lite4_cccf_300_b128_e90_g4](../../configs/cccf/efficientnet_lite4_cccf_300_b128_e90_g4.yaml)   |  efficientnet_lite4  | 66.856 / 88.343 | CrossEntropyLoss |    SGD    |  MultiStepLR |   90  |   False  |
|  [efficientnet_lite1_cccf_240_b256_e90_g4_no_bias-norm](../../configs/cccf/efficientnet_lite1_cccf_240_b256_e90_g4_no_bias-norm.yaml)   |  efficientnet_lite1  | 66.755 / 88.228 | CrossEntropyLoss |    SGD    |  MultiStepLR |   90  |   False  |
|  [efficientnet_lite1_cccf_224_b256_e90_g4_no_bias-norm](../../configs/cccf/efficientnet_lite1_cccf_224_b256_e90_g4_no_bias-norm.yaml)   |  efficientnet_lite1  | 66.272 / 88.207 | CrossEntropyLoss |    SGD    |  MultiStepLR |   90  |   False  |
|  [efficientnet_lite3_cccf_280_b128_e90_g4_no_bias-norm](../../configs/cccf/efficientnet_lite3_cccf_280_b128_e90_g4_no_bias-norm.yaml)   |  efficientnet_lite3  | 66.108 / 87.999 | CrossEntropyLoss |    SGD    |  MultiStepLR |   90  |   False  |
|  [efficientnet_lite4_cccf_224_b128_e90_g4](../../configs/cccf/efficientnet_lite4_cccf_224_b128_e90_g4.yaml)   |  efficientnet_lite4  | 65.926 / 88.156 | CrossEntropyLoss |    SGD    |  MultiStepLR |   90  |   False  |
|  [efficientnet_lite2_cccf_224_b256_e90_g4_no_bias-norm](../../configs/cccf/efficientnet_lite2_cccf_224_b256_e90_g4_no_bias-norm.yaml)   |  efficientnet_lite2  | 65.676 / 87.948 | CrossEntropyLoss |    SGD    |  MultiStepLR |   90  |   False  |
|  [efficientnet_lite3_cccf_224_b256_e90_g4_no_bias-norm](../../configs/cccf/efficientnet_lite3_cccf_224_b256_e90_g4_no_bias-norm.yaml)   |  efficientnet_lite3  | 65.182 / 87.679 | CrossEntropyLoss |    SGD    |  MultiStepLR |   90  |   False  |
|  [efficientnet_lite1_cccf_224_b256_e90_g4_ls_no_bias-norm](../../configs/cccf/efficientnet_lite1_cccf_224_b256_e90_g4_ls_no_bias-norm.yaml)   |  efficientnet_lite1  | 64.469 / 87.092 | CrossEntropyLoss |    SGD    |  MultiStepLR |   90  |   False  |
|  [efficientnet_lite4_cccf_300_b128_e90_g4_mixup](../../configs/cccf/efficientnet_lite4_cccf_300_b128_e90_g4_mixup.yaml)   |  efficientnet_lite4  | 64.469 / 86.536 | CrossEntropyLoss |    SGD    |  MultiStepLR |   90  |   False  |
|  [efficientnet_lite4_cccf_300_b128_e90_g4_no_bias-norm](../../configs/cccf/efficientnet_lite4_cccf_300_b128_e90_g4_no_bias-norm.yaml)   |  efficientnet_lite4  | 62.422 / 85.358 | CrossEntropyLoss |    SGD    |  MultiStepLR |   90  |   False  |
|  [efficientnet_lite4_cccf_224_b128_e90_g4_no_bias-norm](../../configs/cccf/efficientnet_lite4_cccf_224_b128_e90_g4_no_bias-norm.yaml)   |  efficientnet_lite4  | 62.062 / 85.346 | CrossEntropyLoss |    SGD    |  MultiStepLR |   90  |   False  |
|  [efficientnet_lite4_cccf_300_b128_e90_g4_adam](../../configs/cccf/efficientnet_lite4_cccf_300_b128_e90_g4_adam.yaml)   |  efficientnet_lite4  | 62.929 / 86.454 | CrossEntropyLoss |    SGD    |  MultiStepLR |   90  |   False  |
|  [efficientnet_lite4_cccf_300_b128_e90_g4_adam_no_bias-norm](../../configs/cccf/efficientnet_lite4_cccf_300_b128_e90_g4_adam_no_bias-norm.yaml)   |  efficientnet_lite4  | 62.003 / 85.409 | CrossEntropyLoss |    SGD    |  MultiStepLR |   90  |   False  |