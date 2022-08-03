# CHANGE

## v0.5.3

* New features
  1. perf(sampler): when in distribued, set train_sampler be shuffle=True. [80f35daee50e78b](https://github.com/ZJCV/ZCls2/tree/80f35daee50e78bb8e1aac5d41e7eaf6a90dd104)
  2. perf(train.py): add device setting for single GPU training. [932e466f5](https://github.com/ZJCV/ZCls2/tree/932e466f514903a60bb03d66edca9487ecac2671)
  3. perf(model): update DDP(model) use. [c25110795c](https://github.com/ZJCV/ZCls2/tree/c25110795c61ec2ea2346c48f59c8e8242778019)
* Bug fixes
* Breaking changes.

## v0.5.2

* New features
  1. build(python): update requirements.txt. [aa196099a](https://github.com/ZJCV/ZCls2/tree/aa196099aa1067a7f31dffdecb63c3a7b98962e6)
* Bug fixes
* Breaking changes.

## v0.5.1

* New features
  1. build(python): update setup.py INSTALL_REQUIRES. [3ce08a61b1](https://github.com/ZJCV/ZCls2/tree/3ce08a61b1a0f2c0cc2199a25dde902e671ac018)
  2. fix(setup.py): fix INSTALL_REQUIRES settings. [03050ebc](https://github.com/ZJCV/ZCls2/tree/03050ebcde1ba62724bc162d0ad624d070194dc2)
* Bug fixes
* Breaking changes.

## v0.5.0

* New features
  1. perf(train): convert nvidia/apex to torch.cuda.apex. [d165dbc23ad](https://github.com/ZJCV/ZCls2/tree/d165dbc23adb3b659198bc3e6178175973c8def9)
* Bug fixes
  1. fix(tools): update train.sh and eval.sh. [7ca74fc5d86](https://github.com/ZJCV/ZCls2/tree/7ca74fc5d865a203e03301a4230fa42ff1157567)
* Breaking changes.

## v0.4.5

* New features
  1. fix(transform): fix Normalize mean/std with max_value use. [51fc80d2](https://github.com/ZJCV/ZCls2/tree/51fc80d2a4346ae4a027a117fd5186eaf64df61f)
* Bug fixes
* Breaking changes.

## v0.4.4

* New features
  1. fix(cccf.py): fix class_path load. [3ef83e8a7f](https://github.com/ZJCV/ZCls2/tree/3ef83e8a7fafdb3486d1e50d3b287832c66a7b98)
* Bug fixes
* Breaking changes.

## v0.4.3

* New features
  1. perf(dataset): update torchvision.dataset.ImageFolder and build.py use. [e95f4991](https://github.com/ZJCV/ZCls2/tree/e95f49911cc0d6aea24aba86b1ace8795326eea5)
* Bug fixes
* Breaking changes.

## v0.4.2

* New features
  1. feat(transform): add max_value param in _C.TRANSFORM.NORMALIZE. [f51f304](https://github.com/ZJCV/ZCls2/tree/f51f304dadeea33156d7238c32c80146bec051b7)
* Bug fixes
  1. fix(criterion): fix SoftTargetCrossEntropy use. [04cd13c](https://github.com/ZJCV/ZCls2/tree/04cd13cad0ebb7aac34e30b635e56c449f46b54d)
* Breaking changes.

## v0.4.1

* New features
  1. feat(config): add _C.TRAIN.CALCULATE_ACCURACY. [97f3c791](https://github.com/ZJCV/ZCls2/tree/97f3c79181bc766fa4d6d5330c2c1c1cf709bfa9)
* Bug fixes
  1. fix(trainer.py): fix mixup usage. [1140fe47](https://github.com/ZJCV/ZCls2/tree/1140fe477f25bf0c523d01f060e20b1dac8a2938)
* Breaking changes.

## v0.4.0

* New features
  1. perf(misc.py): update resume() to misc.py. [4f417e194](https://github.com/ZJCV/ZCls2/tree/4f417e1941092c8d08c22edf13053e7c231b9790)
  2. perf(parser.py): remove --resume settings. [710791e](https://github.com/ZJCV/ZCls2/tree/710791e63a80c72ea74a742c29d2ef5401c2431d)
  3. feat(tools): add eval.sh. [545e4e4](https://github.com/ZJCV/ZCls2/tree/545e4e41912a6ac0fca33923817d56b0d8a1da0f)
  4. feat(optimizer): add Adam. [7cd0a3a2](https://github.com/ZJCV/ZCls2/tree/7cd0a3a2b8b86351b22680b235ca08ba46a2c0aa)
* Bug fixes
  1. fix(train.py): make mixup_fn to train() and update resume() usage. [b5be2ed3](https://github.com/ZJCV/ZCls2/tree/b5be2ed3526eb3cb3235c9882a604c8931b01059)
* Breaking changes.

## v0.3.0

* New features
  1. style(models): update __all__ use. [0467e383](https://github.com/ZJCV/ZCls2/tree/0467e3837013dd4ba030e58210e7a4fc88828df3)
* Bug fixes
  1. fix(soft_target_cross_entropy_loss.py): set targets to one-hot code. [08352474](https://github.com/ZJCV/ZCls2/tree/08352474c2ad5c7922bfe647a5db2143bd995add)
* Breaking changes
  1. refactor(zcls2): refactor criterion/optimizer/lr_scheduler use. [c740025](https://github.com/ZJCV/ZCls2/tree/c740025f468331ce7423eb108a6e8230165ee4c1)

## v0.2.3

* New features
  1. feat(model): add EfficientNetLite support. [84172a15](https://github.com/ZJCV/ZCls2/tree/84172a15102c3698a951c7c32accab3f263f38c7)
  2. feat(transform): add Mixup + Cutmix support. [902a102e](https://github.com/ZJCV/ZCls2/tree/902a102e28371b459473fb0eb4a078f53fdd69b3)
  3. feat(criterion): add label_smoothing support. [6b4eb4eb](https://github.com/ZJCV/ZCls2/tree/6b4eb4ebdbd534d2b0b35c82b96ca21590174a98)
  4. feat(model): add EfficientNet. [aaebc5b6](https://github.com/ZJCV/ZCls2/tree/aaebc5b6c46f08fed8f27847e51411d317f25f3f)
  5. perf(multi_step_lr.py): add cfg.LR_SCHEDULER.MULTISTEP_LR.STEP_SIZE. [2494defb](https://github.com/ZJCV/ZCls2/tree/2494defb2799a72752bf1e29bc6a0a9fb8bbe851)
  6. feat(optimizer): add RMSPRop optimizer. [b3669030](https://github.com/ZJCV/ZCls2/tree/b3669030159d85cdeb34e9d90bf2c3fb0fa27c38)
* Bug fixes
* Breaking changes

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