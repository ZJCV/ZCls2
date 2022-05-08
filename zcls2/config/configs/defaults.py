from yacs.config import CfgNode as CN

_C = CN()

# Output basedir.
_C.OUTPUT_DIR = "./outputs/tmp"

# Note that non-determinism may still be present due to non-deterministic
# operator implementations in GPU operator libraries.
_C.RNG_SEED = 1

_C.DETERMINISTIC = False

# Only run 10 iterations for profiling.
_C.PROF = -1

# resume model weights, train epoch, criterion, optimizer and lr_scheduler
_C.RESUME = ""

# evaluate model on validation set
_C.EVALUATE = False

_C.CHANNELS_LAST = False

# print frequency (default: 10)
_C.PRINT_FREQ = 10

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

# number of epoch to begin train (default: 1)
_C.TRAIN.START_EPOCH = 1

# how many rounds to use model infer test dataset
_C.TRAIN.EVAL_EPOCH = 1

# number of total epochs to run
_C.TRAIN.MAX_EPOCH = 90

# Specify values of k for the precision@k
_C.TRAIN.TOP_K = (1, 5)

# Perform accuracy calculation in the training phase
# Applicable to classification model
_C.TRAIN.CALCULATE_ACCURACY = True