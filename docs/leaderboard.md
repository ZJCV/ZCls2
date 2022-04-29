
# LeadBoard(Based on CCCF)

| cfg |    model   |   top1/top5   |       loss       | optimizer | lr-scheduler | epoch |
|:---:|:----------:|:-------------:|:----------------:|:---------:|:------------:|:-----:|
|  [mbv3_large_cccf_224_b256_e90_g4](../configs/cccf/mbv3_large_cccf_224_b256_e90_g4.yaml)   | mbv3_large | 79.56 / 94.90 | CrossEntropyLoss |    SGD    |  MultiStepLR |   90  |
|  [r50_cccf_224_b256_e90_g4](../configs/cccf/r50_cccf_224_b256_e90_g4.yaml)   |  resnet50  | 77.11 / 93.93 | CrossEntropyLoss |    SGD    |  MultiStepLR |   90  |
|     |            |               |                  |           |              |       |
