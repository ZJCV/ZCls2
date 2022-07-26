# Get started

## CONFIGS

make config file like `configs/***.yaml`

## TRAIN

run `train.py` like

```shell
# specify config-path
CUDA_VISIBLE_DEVICES=0,1,2,3 bash tools/train.sh configs/cccf/r50_cccf_224_b256_e90_g4.yaml

# specify config-path and master-port
CUDA_VISIBLE_DEVICES=0,1,2,3 bash tools/train.sh configs/cccf/r50_cccf_224_b256_e90_g4.yaml 31222
```

## EVAL

If you want to eval only. Do it like

```shell
# specify config-path
CUDA_VISIBLE_DEVICES=0,1,2,3 bash tools/eval.sh configs/cccf/ghostnet_100_cccf_224_b256_e90_g4.yaml

# specify config-path and master-port
CUDA_VISIBLE_DEVICES=0,1,2,3 bash tools/eval.sh configs/cccf/ghostnet_100_cccf_224_b256_e90_g4.yaml 31226
```