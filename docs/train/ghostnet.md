
# GhostNet

| cfg |    model   |   top1/top5   |       loss       | optimizer | lr-scheduler | epoch | pretrained |
|:---:|:----------:|:-------------:|:----------------:|:---------:|:------------:|:-----:|:-----:|
|  [ghostnet_100_cccf_600_b128_e90_g4](../../configs/cccf/ghostnet_100_cccf_600_b128_e90_g4.yaml)   | ghostnet_100 | 82.291 / 96.024 | CrossEntropyLoss |    SGD    |  MultiStepLR |   90  |   True  |
|  [ghostnet_100_cccf_300_b256_e90_g4](../../configs/cccf/ghostnet_100_cccf_300_b256_e90_g4.yaml)   | ghostnet_100 | 80.972 / 95.460 | CrossEntropyLoss |    SGD    |  MultiStepLR |   90  |   True  |
|  [ghostnet_100_cccf_224_b256_e90_g4](../../configs/cccf/ghostnet_100_cccf_224_b256_e90_g4.yaml)   | ghostnet_100 | 79.801 / 95.00 | CrossEntropyLoss |    SGD    |  MultiStepLR |   90  |   True  |
|  [ghostnet_100_cccf_224_b256_e90_g4_ft](../../configs/cccf/ghostnet_100_cccf_224_b256_e90_g4_ft.yaml)   | ghostnet_100 | 78.373 / 94.813 | CrossEntropyLoss |    SGD    |  MultiStepLR |   90  |   True  |
|  [ghostnet_130_cccf_224_b256_e90_g4](../../configs/cccf/ghostnet_130_cccf_224_b256_e90_g4.yaml)   |  ghostnet_130  | 72.151 / 91.706 | CrossEntropyLoss |    SGD    |  MultiStepLR |   90  |   False  |
|  [ghostnet_130_cccf_224_b256_e90_g4_v2](../../configs/cccf/ghostnet_130_cccf_224_b256_e90_g4_v2.yaml)   |  ghostnet_130  | 70.089 / 90.507 | CrossEntropyLoss |    SGD    |  MultiStepLR |   90  |   False  |
