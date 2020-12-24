# Visual-Recognition-Final-Project
NCTU Visual Recognition Final Project

## Hardware
OS: Ubuntu 18.04.3 LTS

CPU: Intel(R) Xeon(R) W-2133 CPU @ 3.60GHz

GPU: 1x GeForce RTX 2080 TI

## Reproducing Submission
To reproduct my submission without retrainig, do the following steps:
1. [Installation](#installation)
2. [Dataset Preparation](#Dataset-Preparation)
3. [Training detail](#Training)
4. [Testing detail](#Testing)
5. [Reference](#Reference)

## Installation

this code was trained and tested with Python 3.6.10 and Pytorch 1.3.0 (Torchvision 0.4.1) on Ubuntu 18.04

```
conda create -n finalproject python=3.6
conda activate finalproject
pip install -r requirements.txt
```

## Dataset Preparation
```
cs-t0828-2020-Final
├── Final
│   ├── train_images
│   │   ├── Put all images here
│   ├── train.csv
│   ├── test.csv
│   ├── 2015_train.csv

```
I seperate the original training data (38788 images) into two part. One for training (37626 images) and one for evaluating(1162 images). 

## Training
To train models:

Open the **model.py** with your own IDE and directly run it. 
There are several hyperparameters in the code **276 ~ 295**.

The expected training times are:
Model | GPUs | Image size | Training Epochs | Training Time
------------ | ------------- | ------------- | ------------- | -------------
efficientnet | 1x RTX 2080Ti | 288 x 288 | 50 | 6 hours

*  model_state = "train"
*  batch_size = 16
*  network = 5
*  fc_out_num = 6
*  optimizers = 1
*  initial_lr = 1e-4
*  dynamic_lr = True



## Testing

[Pretrain weight](https://drive.google.com/file/d/1-hFy7fqNaAebOEdYS0bwNdrt9WRWRSru/view?usp=sharing)


To test models:

Open the **model.py** with your own IDE and directly run it. 
There are several hyperparameters in the code **276 ~ 295**.

*  model_state = "eval" 
*  batch_size = 25
*  network = 8
*  ckpt_path = "/PATH/TO/YOUR/WEIGHT/FILE"
*  model_weight = ""epoch_XX.pkl""
*  fc_out_num = 6
*  optimizers = 1
*  initial_lr = 1e-4

## Reference
1. [Efficientnet](https://github.com/lukemelas/EfficientNet-PyTorch).
2. [Kaggle Discussion](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion).
