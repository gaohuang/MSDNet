# MSDNet

This repository provides the code for the paper [Multi-Scale Dense Networks for Resource Efficient Image Classification](http://arxiv.org/abs/1703.09844).

### Update on April 3, 2019 -- PyTorch implementation released!
[A PyTorch implementation of MSDNet can be found from here.](https://github.com/kalviny/MSDNet-PyTorch)


## Introduction

This paper studies convolutional networks that require limited computational resources at test time. We develop a new network architecture that performs on par with state-of-the-art convolutional networks, whilst facilitating prediction in two settings: (1) an anytime-prediction setting in which the network's prediction for one example is progressively updated, facilitating the output of a prediction at any time; and (2) a batch computational budget setting in which a fixed amount of computation is available to classify a set of examples that can be spent unevenly across 'easier' and 'harder' examples. 

**Figure 1**: MSDNet layout (2D).

<img src="https://cloud.githubusercontent.com/assets/6460942/26835456/61a77e08-4aa6-11e7-82be-7eb09a765d7c.png" width="650">

**Figure 2**: MSDNet layout (3D).

<img src="https://cloud.githubusercontent.com/assets/6460942/26835513/6daa5630-4aa6-11e7-999d-4d4e0c372105.png" width="650">


## Results
### (a) anytime-prediction setting 

**Figure 3**: Anytime prediction on ImageNet.

<img src="https://cloud.githubusercontent.com/assets/16090466/24482636/e00c3ad4-14c0-11e7-93b6-cb7a6feb7634.png" width="400">


### (b) batch computational budget setting

**Figure 4**: Prediction under batch computational budget on ImageNet.

<img src="https://cloud.githubusercontent.com/assets/16090466/24482632/da038f16-14c0-11e7-8864-ca5c20bcafde.png" width="400">

**Figure 5**: Random example images from the ImageNet classes `Red wine` and `Volcano`. Top row: images exited from the first classification layer of an MSDNet with correct prediction; Bottom row: images failed to be correctly classified at the first classifier but were correctly predicted and exited at the last layer.

<img src="https://cloud.githubusercontent.com/assets/16090466/24482639/e5666626-14c0-11e7-99d3-5f39a2a631ac.png" width="400">


## Usage

Our code is written under the framework of Torch ResNet (https://github.com/facebook/fb.resnet.torch). The training scripts come with several options, which can be listed with the `--help` flag.

    th main.lua --help

#### Configuration

In all the experiments, we use a **validation set** for model selection. We hold out `5000` training images on CIFAR, and 
`50000` 
images on ImageNet as the validation set.


#### Training recipe

Train an MSDNet with 10 classifiers attached to every other layer for anytime prediction:
```bash
th main.lua -netType msdnet -dataset cifar10 -batchSize 64 -nEpochs 300 -nBlocks 10 -stepmode even -step 2 -base 4
```

Train an MSDNet with 7 classifiers with the span linearly increases for efficient batch computation:
```bash
th main.lua -netType msdnet -dataset cifar10 -batchSize 64 -nEpochs 300 -nBlocks 7 -stepmode lin_grow -step 1 -base 1
```

#### Pre-trained ImageNet Models

1. [Download](https://www.dropbox.com/sh/elnyjl4xdi4zyas/AACxCdjV-RWYrHYfbz61FFDma?dl=0) model checkpoints and the validation set indeces.
	
2. Testing script: `th main.lua -dataset imagenet -testOnly true -resume <path-to-.t7-model> -data <path-to-image-net-data> -gen <path-to-validation-set-indices>`

## FAQ

1. How to calculate the FLOPs (or mul-add op) of a model?

We strongly recommend doing it automatically. Please refer to the [op-counter](https://github.com/apaszke/torch-opCounter) project (LuaTorch), or the [script](https://github.com/ShichenLiu/CondenseNet/blob/master/utils.py#L58-L162) in ConDenseNet (PyTorch). The basic idea of these op counters is to add a hook before the forward pass of a model.






