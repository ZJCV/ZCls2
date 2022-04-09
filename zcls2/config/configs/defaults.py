from yacs.config import CfgNode as CN

_C = CN()

# Output basedir.
_C.OUTPUT_DIR = "./outputs/tmp"

# Note that non-determinism may still be present due to non-deterministic
# operator implementations in GPU operator libraries.
_C.RNG_SEED = 1

_C.DETERMINISTIC = False

# ---------------------------------------------------------------------------- #
# Distributed options
# ---------------------------------------------------------------------------- #
_C.DISTRIBUTED = False

# Number of GPUs to use (applies to both training and testing).
_C.NUM_GPUS = 1

# Number of machine to use for the job.
_C.NUM_NODES = 1

# The index of the current machine.
_C.RANK_ID = 0

# Distributed backend.
_C.DIST_BACKEND = "nccl"

# Initialization method, includes TCP or shared file-system
_C.INIT_METHOD = "env://"

# ---------------------------------------------------------------------------- #
# Train
# ---------------------------------------------------------------------------- #
_C.TRAIN = CN()

# Only run 10 iterations for profiling.
_C.PROF = -1

# print frequency (default: 10)
_C.PRINT_FREQ = 10

# refert to
# [How to Break GPU Memory Boundaries Even with Large Batch Sizes](https://towardsdatascience.com/how-to-break-gpu-memory-boundaries-even-with-large-batch-sizes-7a9c27a400ce)
# [How to implement accumulated gradient？](https://discuss.pytorch.org/t/how-to-implement-accumulated-gradient/3822)
_C.TRAIN.GRADIENT_ACCUMULATE_STEP = 1

# number of epoch to begin train
_C.TRAIN.START_EPOCH = 0

# how many rounds to save training params, includes model weights, train epoch, criterion, optimizer and lr_scheduler
_C.TRAIN.SAVE_EPOCH = 1

# how many rounds to use model infer test dataset
_C.TRAIN.EVAL_EPOCH = 1

# number of total epochs to run
_C.TRAIN.MAX_EPOCH = 90

# resume model weights, train epoch, criterion, optimizer and lr_scheduler
_C.RESUME = ""

# evaluate model on validation set
_C.EVALUATE = False

# note: when using clip_gradient, set too small MAX_NORM value will make training slower
# 1. [Proper way to do gradient clipping?](https://discuss.pytorch.org/t/proper-way-to-do-gradient-clipping/191)
# 2. [pytorch/torch/nn/utils/clip_grad.py](https://github.com/pytorch/pytorch/blob/master/torch/nn/utils/clip_grad.py)
# 3. [How to do gradient clipping in pytorch?](https://stackoverflow.com/questions/54716377/how-to-do-gradient-clipping-in-pytorch)
_C.TRAIN.CLIP_GRADIENT = False
_C.TRAIN.MAX_NORM = 20.0