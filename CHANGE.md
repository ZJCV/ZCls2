# CHANGE

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