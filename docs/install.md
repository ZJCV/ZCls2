
# Install

## Requirements


    Linux (Windows is not officially supported)
    Python 3.8
    PyTorch 1.8.2 or higher
    torchvison 0.9.2 or higher

About `apex`，see [NVIDIA/apex](https://github.com/NVIDIA/apex). I recommend people to use docker training environment, see

* [NGC Pytorch](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch)
* [PyTorch Release Notes](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/index.html)

## Install ZCls2

### Source install

* Install `pytorch/torchvision/apex` following the official instructions.

* Clone the `ZCls2` repository.

```
git clone https://github.com/ZJCV/ZCls2.git
cd ZCls2
```

* Install `ZCls2`.

```
python3 setup.py install
```

### Pip install (*RECOMMEND*)

```python
pip3 install zcls2
```