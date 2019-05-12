# Content

[PyTorch](https://pytorch.org) version to Image Captioning.
> The baseline version use [sgrvinod](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning)

> The version one use the residual attention module

> The version two use faster rcnn and merge module to generate words

I'm using `PyTorch 0.4` in `Python 3.6`.

## Installation
- [PyTorch 0.4](https://pytorch.org)
- [nltk 3.2.5](http://www.nltk.org/)
- [h5py 2.7.1](http://www.h5py.org/)
- [tqdm 4.29.1](https://pypi.org/project/tqdm/)
- [numpy 1.14.0](http://www.numpy.org/)
- [scipy 1.1.0](https://www.scipy.org/)
- [matplotlib 2.2.1](https://matplotlib.org/)
- [seaborn 0.8.1](http://seaborn.pydata.org/)

## Introducation
### baseline version
The baseline version is based on [show attend and tell](https://arxiv.org/abs/1502.03044)

The detail of the implementation can see [README](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning/blob/master/README.md)

### V1
The version one is based on residual attention block,which is an important part of [residual attention network](https://arxiv.org/abs/1704.06904).

### V2
The version two combined [Faster RCNN](https://arxiv.org/abs/1506.01497) with a caption generation network based on "merge" module.


## Implementation
The sections below briefly describe the implementation

### Dataset

I'm using the MSCOCO '14 Dataset. You'd need to download the [Training (13GB)](http://images.cocodataset.org/zips/train2014.zip) and [Validation (6GB)](http://images.cocodataset.org/zips/val2014.zip) images.

We will use [Andrej Karpathy's training, validation, and test splits](http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip). This zip file contain the captions. You will also find splits and captions for the Flicker8k and Flicker30k datasets, so feel free to use these instead of MSCOCO if the latter is too large for your computer.
***
### Training

See [`train.py`]()

To **train your model from scratch**, simply run this file 
> ```python train.py```
***
### Eval

See [`eval.py`]()

It will compute the correct BLEU-4 scores of model checkpoint on the test set
> ```python eval.py```

