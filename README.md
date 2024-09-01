OASG-Net: Occlusion Aware and Structure-Guided Network for Face De-Occlusion
=================================
This is the PyTorch implementation of paper 'OASG-Net: Occlusion Aware and Structure-Guided Network for Face De-Occlusion'.

Introduction
---------------------------------
This paper studies an effective deep learning-based strategy for masked face restoration. The network consists of a mask prediction subnet, a facial landmark prediction subnet and an face restoration subnet. The mask prediction subnet is used to predict the mask position in a masked image, the facial landmark prediction subnet aims to predict the key points of the face covered by the mask as additional information for the image restoration subnet, and the restoration subnet generates a credible appearance based on the given predicted landmarks.

![image](lafin.png)

Prerequisites
---------------------------------
* Python 3.7
* Pytorch 1.0
* opencv_python
* scipy==1.2.0
* scikit_image==0.14.2
* torchvision==0.2.1
* torch
* matplotlib==3.0.3
* numpy==1.15.4
* face_alignment==1.0.1
* Pillow==6.2.1
* PyYAML==5.1.2
* skimage==0.0


Installation
---------------------------------
* Clone this repo:
```
git clone https://github.com/YaN9-Y/lafin
cd lafin-master
```
* Install Pytorch
* Install python requirements:
```
pip install -r requirements.txt
```

Datasets
---------------------------------
Our repo has three parts, 1) Mask Prediction Part and 2)Image Inpainting Part and 3)Augmented Landmark Detection Part. 

### 1.Mask Prediction Part
#### 1) Images: 
We use [masekd_whn dataset](https://github.com/X-zhangyang/Real-World-Masked-Face-Dataset) datasets for mask prediction. We have marked the positions of the masks. The link to the marked dataset is (https://pan.baidu.com/s/1pA_qNthVHsKS24u8MpQbRg?pwd=e0td) extraction code：e0td.


### 2.Image Inpaint Part
#### 1) Images: 

We use [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) and [CelebA-HQ](https://github.com/tkarras/progressive_growing_of_gans) datasets for inpaint. and you can use this tool(https://github.com/aqeelanwar/MaskTheFace) to generate more synthetic mask image in your training dataset.

After downloading, you should split the whole dataset to train, test and validation set and run `scripts/flist.py` to genrate corresponding file lists. For example, to generate the training set file list on the CelebA dataset, you should run:
```
mkdir datasets
python3 ./scripts/flist.py --path path_to_celebA_train_set --output ./datasets/celeba_train_images.flist
```

For CelebA-HQ dataset, we use its 256x256 version. For CelebA dataset, the original image will be center cropped then resized to 256x256 during training.

#### 2) Landmarks:
For CelebA and CelebA-HQ datasets, the landmarks given by the original dataset are not enough (only 5). So we apply [FAN](https://github.com/1adrianb/face-alignment) to generate landmarks as ground truth landmarks.

You can run `scripts/preprocess_landmark.py` to generate landmarks of the images, then use `scripts/flist.py` to generate landmarks file list. For example, to generate the landmarks of the CelebA training images, you should run:
```
python3 ./scripts/preprocess_landmark.py --path path_to_CelebA_train_set --output path_to_the_celebA_landmark_train_set
python3 ./scripts/flist.py --path path_to_celebA_landmark_train_set --output ./datasets/celeba_train_landmarks.flist
```
This may take a while.

#### 3) Masks:
you can use block masks, irregular masks or synthetic Mask given by MaskTheFace to train your model. The irregular masks dataset provided by [Liu et al.](https://arxiv.org/abs/1804.07723) is available on [their website](http://masc.cs.gmu.edu/wiki/partialconv)

Then use `scripts/flist.py` to generate train/test/validataion masks file lists as above.

### 3.Augmented Landmark Detection Part
To validate the landmark detection augmented by inpainted images, please firstly download [WFLW](https://wywu.github.io/projects/LAB/WFLW.html) dataset provided by Wu et al.. 

After downloading, run `scripts/preprocess_wflw.py` to generate train/test/validation images and landmarks then run `scripts/flist.py` to generate train/test file lists.
```
python3 ./scripts/preprocess_wflw.py --path  path_to_the_WFLW_images_folder --output path_to_the_output_folder --annotation_path path_to_the_WFLW_annotations_folder
python3 ./scripts/flist.py --path path_to_the_wflw_train/test_images/landmarks_folder --output ./datasets/wflw_train/test_images/landmarks.flist 
```

Getting Started
--------------------------
To use the pre-trained models, download them from the following links then copy them to corresponding checkpoints folder, like `./checkkpoints/celeba` or `./checkpoints/celeba-hq`.

[CelebA](https://drive.google.com/open?id=1lGFEbxbtZwpPA9JXF-bhv12Tdi9Zt08G) | [CelebA-HQ](https://drive.google.com/open?id=1Xwljrct3k75_ModHCkwcNjJk3Fsvv-ra) | [WFLW](https://drive.google.com/open?id=1I2MzHre1U3wqTu5ZmGD36OiXPaNqlOKb)

### 1.Mask Prediction Part
#### 1) Testing
The train code of Mask Prediction Part is missed, but the model file and prediction file is remained, you can use the model code and pretrained model to Test it run './script/PredictTheMask.py' to predict the mask of a mask face.

```
python3 script/PredictTheMask.py --path 1.png --output result.png
```

#### 2) Training
you can use (https://pan.baidu.com/s/1pA_qNthVHsKS24u8MpQbRg?pwd=e0td) to finish a training script your self. The mask of any picture is given by the same name .json file.

### 2.Image Inpaint Part

#### 1) Training 
To train the model, create a `config.yml` file similar to `config.yml.example` and copy it to corresponding checkpoint folder. Following comments on `config.yml.example` to set `config.yml`.

The inpaint model is trained in two stages: 1) train the landmark prediction model, 2) train the image inpaint model. To train the model, run:

```
python train.py --model [stage] --checkpoints [path to checkpoints]
``` 

For example, to train the landmark prediction model on CelebA dataset, the checkpoints folder is `./checkpoints/celeba` folder, run:

```
python3 train.py --model 1 --checkpoints ./checkpoints/celeba
```

The number of training iterations can be changed by setting `MAX_ITERS` in `config.yml`.

### 3.Augmented Landmark Detection Part
#### 1) Training
We suppose you use WFLW dataset to validate the augmented landmark detection method.
To validate the augmentation methods, a landmark-guided inpaint model trained on WFLW (stage 2) is needed. You can train it by yourself following above steps or use the pre-trained models.

Create a `config.yml` file similar to `config.yml.example` and copy it to corresponding checkpoint folder. Following comments on `config.yml.example` to set `config.yml`.
Remeber set `AUGMENTATION_TRAIN = 1` to enable augmentation with inpainted images, amd `LANDMARK_POINTS = 98` in `config.yml`.
Then run:
```
python3 train.py --model 1 --checkpoints ./checkpoints/wflw
```
to start augmentated training.
# OASG-Net-Occlusion-Aware-and-Structure-Guided-Network-for-Face-De-Occlusion
