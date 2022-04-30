# CHANGE

## v0.2.2

* New features
  1. build(python): update timm version and change >= to ~=. [80cd4a38](https://github.com/ZJCV/ZCls2/tree/80cd4a38f2a83e9223cebe7a3b890c8e4a44ea3e)
  2. perf(train.sh): make CUDA_VISIBLE_DEVICES and master_port configurable. [c5385bf7](https://github.com/ZJCV/ZCls2/tree/c5385bf7f7c6b9406989231ce53584c781adb53a)
* Bug fixes
  1. fix(model): fix ghostnet replacing the forward function failed. [48794a1d](https://github.com/ZJCV/ZCls2/tree/48794a1de822ac48d58d41213d04eebdf6111d5d)
  2. fix(cccf): convert gray img to three-channel. [96fb0fe3](https://github.com/ZJCV/ZCls2/tree/96fb0fe36cbbf168201c5623a784aa99d722a00e)
  3. fix(infer): fix wrong use param i. [820b4e2d](https://github.com/ZJCV/ZCls2/tree/820b4e2d01be88e81053cbbe177f6dde9802618f)
  4. fix(trainer): fix wrong use param i. [d503e227](https://github.com/ZJCV/ZCls2/tree/d503e2276b912105a50df70feaf6dd0c46556dde)
* Breaking changes

## v0.2.1

* New features
  1. feat(dataset): new Dataset CCCF. [d02b2f66](https://github.com/ZJCV/ZCls2/tree/d02b2f66ec56c57dcf2345639d7ca423ef4abe8a)
     1. The CCCF is a custom mixed classification dataset
     2. Including CIFAR100/CUB-200-2011/Caltech-101/Food-101
  2. feat(configs): add cfg.TRAIN.TOP_K supports and reformat prec@k log. [7877607f](https://github.com/ZJCV/ZCls2/tree/7877607fe3d274bd014edfb9a6b56b96ccb70de3)
  3. chore(benchmarks): add mobilenetv2 for cifar10/cifar100/fashionmnist. [ef753ed5](https://github.com/ZJCV/ZCls2/tree/ef753ed538dd353a7a6472f8d8fe1e8ec5929b25)
* Bug fixes
* Breaking changes

## v0.2.0

* New features
  1. Update benchmarks (`Apex` vs. `ZCls2`)
* Bug fixes
* Breaking changes
  1. Change model output type. [899c725](https://github.com/ZJCV/ZCls2/commit/899c725655a59ec09d5cdb043b4ebb3f7c05eea6)

## v0.1.1

* New features
  1. Adjust `zcls2` python version constraints (`python >= 3.9` to `python >= 3.8`)
* Bug fixes
* Breaking changes.

## v0.1.0

* New features
    1. New-built training module, default supported (derived from `nvidia/apex`)
       1. `Distributed training`
       2. `Mixed-precision training`
       3. `Linear warmup`
       4. `Data prefetcher`
    2. New-built model module, support `resnet(torchvision)/mobilenet(torchvision)/ghostnet(timm)`
    3. New-built criterion module, support `CrossEntroyLoss/LargeMarginSoftmaxLoss`
    4. New-built optimizer module, support `SGD`
    5. New-built lr_scheduler module, support `MultiStepLR/CosineAnnealingLR`
    6. New-built dataset module, support
       1. `CIFAR10/CIFAR100`
       2. `GeneralDataset/GeneralDatasetV2`
       3. `MPDataset`
    7. New-built transform module, support (derived from `torchvision`)
       1. Normal transform (`ConvertImageDtype/Normalize/ToPILImage/ToTensor`)
       2. Color transform (`ColorJitter/Grayscale/RandomAutocontrast/RandomAutocontrast/RandomAdjustSharpness/RandomErasing/RandomPosterize`)
       3. Geometric transform (`CenterCrop/RandomCrop/RandomHorizontalFlip/RandomVerticalFlip/RandomRotation/RandomResizedCrop/Resize`)
       4. Augment (`AutoAugment/RandAugment`)
       5. Custom (`OpenCVResize/SquarePad`)
    8. New-built config module (derived from `ZJCV/ZCls`)
    9. New-built logging module (derived from `ZJCV/ZCls`)
* Bug fixes
* Breaking changes.