# MSDNet

This repository provides the code for the paper [Multi-Scale Dense Convolutional Networks for Efficient Prediction](http://arxiv.org/abs/1703.09844).


## Introduction

This paper studies convolutional networks that require limited computational resources at test time. We develop a new network architecture that performs on par with state-of-the-art convolutional networks, whilst facilitating prediction in two settings: (1) an anytime-prediction setting in which the network's prediction for one example is progressively updated, facilitating the output of a prediction at any time; and (2) a batch computational budget setting in which a fixed amount of computation is available to classify a set of examples that can be spent unevenly across 'easier' and 'harder' examples. 

Figure 1: MSDNet layout.

<img src="https://cloud.githubusercontent.com/assets/16090466/24482563/3bd3d224-14c0-11e7-98d8-cb4b39be6ad9.png" width="400">

## Results
### (a) anytime-prediction setting 

Figure 2: Anytime prediction on ImageNet.

<img src="https://cloud.githubusercontent.com/assets/16090466/24482636/e00c3ad4-14c0-11e7-93b6-cb7a6feb7634.png" width="400">


### (b) batch computational budget setting

Figure 3: Prediction under batch computational budget on ImageNet.

<img src="https://cloud.githubusercontent.com/assets/16090466/24482632/da038f16-14c0-11e7-8864-ca5c20bcafde.png" width="400">

Figure 4: Random example images from the ImageNet classes \emph{Red wine} and \emph{Volcano}. Top row: images exited from the first classification layer of a \methodnameshort{} with correct prediction; Bottom row: images failed to be correctly classified at the first classifier but were correctly predicted and exited at the last layer.

<img src="https://cloud.githubusercontent.com/assets/16090466/24482639/e5666626-14c0-11e7-99d3-5f39a2a631ac.png" width="400">


## Usage

Our code is written under the framework of Torch ResNet (https://github.com/facebook/fb.resnet.torch). The training scripts come with several options, which can be listed with the `--help` flag.

    `th main.lua --help`

#### Configuration

In all the experiments, we use a **validation set** for model selection. We hold out `5000` training images on CIFAR, and 
`50000` 
images on ImageNet as the validation set.


#### Training recipe

Train an MSDNet with 11 classifiers attached to every other layer for anytime prediction:
```bash
th main.lua -netType msdnet -dataset cifar10 -batchSize 64 -nEpochs 300 -nBlocks 11 -stepmode even -step 2 -base 4
```

Train an MSDNet with 7 classifiers with the span linearly increases for efficient batch computation:
```bash
th main.lua -netType msdnet -dataset cifar10 -batchSize 64 -nEpochs 300 -nBlocks 7 -stepmode lin_grow -step 1 -base 1
```
