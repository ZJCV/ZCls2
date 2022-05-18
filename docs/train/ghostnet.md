
# GhostNet

| cfg |    model   |   top1/top5   |       loss       | optimizer | lr-scheduler | epoch | pretrained |
|:---:|:----------:|:-------------:|:----------------:|:---------:|:------------:|:-----:|:-----:|
|  [ghostnet_100_cccf_600_b128_e90_g4](../../configs/cccf/ghostnet_100_cccf_600_b128_e90_g4.yaml)   | ghostnet_100 | 82.291 / 96.024 | CrossEntropyLoss |    SGD    |  MultiStepLR |   90  |   True  |
|  [ghostnet_100_cccf_300_b256_e90_g4_ra](../../configs/cccf/ghostnet_100_cccf_300_b256_e90_g4_ra.yaml)   | ghostnet_100 | 81.197 / 95.690 | CrossEntropyLoss |    SGD    |  MultiStepLR |   90  |   True  |
|  [ghostnet_100_cccf_300_b256_e90_g4_no_bias-norm](../../configs/cccf/ghostnet_100_cccf_300_b256_e90_g4_no_bias-norm.yaml)   | ghostnet_100 | 80.972 / 95.460 | CrossEntropyLoss |    SGD    |  MultiStepLR |   90  |   True  |
|  [ghostnet_100_cccf_300_b256_e90_g4](../../configs/cccf/ghostnet_100_cccf_300_b256_e90_g4.yaml)   | ghostnet_100 | 80.498 / 95.236 | CrossEntropyLoss |    SGD    |  MultiStepLR |   90  |   True  |
|  [ghostnet_100_cccf_224_b256_e90_g4_no_bias-norm](../../configs/cccf/ghostnet_100_cccf_224_b256_e90_g4_no_bias-norm.yaml)   | ghostnet_100 | 79.801 / 95.00 | CrossEntropyLoss |    SGD    |  MultiStepLR |   90  |   True  |
|  [ghostnet_100_cccf_224_b256_e90_g4_ra](../../configs/cccf/ghostnet_100_cccf_224_b256_e90_g4_ra.yaml)   | ghostnet_100 | 79.542 / 95.005 | CrossEntropyLoss |    SGD    |  MultiStepLR |   90  |   True  |
|  [ghostnet_100_cccf_224_b256_e90_g4](../../configs/cccf/ghostnet_100_cccf_224_b256_e90_g4.yaml)   | ghostnet_100 | 78.981 / 94.675 | CrossEntropyLoss |    SGD    |  MultiStepLR |   90  |   True  |
|  [ghostnet_100_cccf_224_b256_e90_g4_ft_ra](../../configs/cccf/ghostnet_100_cccf_224_b256_e90_g4_ft_ra.yaml)   | ghostnet_100 | 78.626 / 94.853 | CrossEntropyLoss |    SGD    |  MultiStepLR |   90  |   True  |
|  [ghostnet_100_cccf_224_b256_e90_g4_ft_no_bias-norm](../../configs/cccf/ghostnet_100_cccf_224_b256_e90_g4_ft_no_bias-norm.yaml)   | ghostnet_100 | 78.373 / 94.813 | CrossEntropyLoss |    SGD    |  MultiStepLR |   90  |   True  |
|  [ghostnet_130_cccf_224_b256_e90_g4](../../configs/cccf/ghostnet_130_cccf_224_b256_e90_g4.yaml)   |  ghostnet_130  | 72.151 / 91.706 | CrossEntropyLoss |    SGD    |  MultiStepLR |   90  |   False  |
|  [ghostnet_130_cccf_224_b256_e90_g4_v2](../../configs/cccf/ghostnet_130_cccf_224_b256_e90_g4_v2.yaml)   |  ghostnet_130  | 70.089 / 90.507 | CrossEntropyLoss |    SGD    |  MultiStepLR |   90  |   False  |
