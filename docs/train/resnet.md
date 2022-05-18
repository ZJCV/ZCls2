
# ResNet

| cfg |    model   |   top1/top5   |       loss       | optimizer | lr-scheduler | epoch | pretrained |
|:---:|:----------:|:-------------:|:----------------:|:---------:|:------------:|:-----:|:-----:|
|  [r50_cccf_224_b256_e90_g4_calr](../../configs/cccf/r50_cccf_224_b256_e90_g4_calr.yaml)   |  resnet50  | 80.101 / 95.979 | CrossEntropyLoss |    SGD    |  CosineAnnealingLR |   90  |   True  |
|  [r50_cccf_224_b256_e90_g4_no_bias-norm](../../configs/cccf/r50_cccf_224_b256_e90_g4_no_bias-norm.yaml)   |  resnet50  | 77.11 / 93.93 | CrossEntropyLoss |    SGD    |  MultiStepLR |   90  |   True  |
|  [r50_cccf_224_b256_e90_g4](../../configs/cccf/r50_cccf_224_b256_e90_g4.yaml)   |  resnet50  | 76.999 / 93.780 | CrossEntropyLoss |    SGD    |  MultiStepLR |   90  |   True  |
