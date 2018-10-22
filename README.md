# Switchable Normalization for Semantic Segmentation



This repository contains the code of using Swithable Normalization (SN) in semantic image segmentation, proposed by the paper 
["Differentiable Learning-to-Normalize via Switchable Normalization"](https://arxiv.org/abs/1806.10779).

This is the implementations of the experiments presented in the above paper by using open-source semantic segmentation framework [Scene Parsing on MIT ADE20K](https://github.com/CSAILVision/semantic-segmentation-pytorch).

## Update

- 2018/9/26: The code and trained models of semantic segmentation on ADE20K by using SN are released !
- More results and models will be released soon. 

## Citation

You are encouraged to cite the following paper if you use SN in research or wish to refer to the baseline results.

```
@article{SwitchableNorm,
  title={Differentiable Learning-to-Normalize via Switchable Normalization},
  author={Ping Luo and Jiamin Ren and Zhanglin Peng},
  journal={arXiv:1806.10779},
  year={2018}
}
```

## Getting Started

Use git to clone this repository:

```
git clone https://github.com/switchablenorms/SwitchNorm_Segmentation.git
```

### Environment

The code is tested under the following configurations.

- Hardware: 1-8 GPUs (with at least 12G GPU memories)
- Software: CUDA 9.0, Python 3.6, PyTorch 0.4.0, tensorboardX

### Installation & Data Preparation

Please check the [Environment](https://github.com/CSAILVision/semantic-segmentation-pytorch/blob/master/README.md#environment), [Training](https://github.com/CSAILVision/semantic-segmentation-pytorch/blob/master/README.md#training) and [Evaluation](https://github.com/CSAILVision/semantic-segmentation-pytorch/blob/master/README.md#evaluation) subsection in the repo [Scene Parsing on MIT ADE20K](https://github.com/CSAILVision/semantic-segmentation-pytorch) for a quick start.

### Pre-trained Models

Download SN based ImageNet pretrained model and put them into the `{repo_root}/pretrained_sn`.

#### ImageNet pre-trained models

The backbone models with SN pretrained on ImageNet are available in the format used by above Segmentation Framework and this repo.

- ResNet50v1+SN(8,2)  [[pretrained_SN(8,2)](https://drive.google.com/file/d/1tHJiCZ3CBXJGiIc9b634S4Rd9KLOfr1P/view?usp=sharing)]


For more pretrained models with SN, please refer to the repo of [switchablenorms/Switchable-Normalization](https://github.com/switchablenorms/Switchable-Normalization).
The following script converts the model trained from [Switchable-Normalization](https://github.com/switchablenorms/Switchable-Normalization) into a valid format used by the semantic segmentation codebase :  `./pretrained_sn/convert_sn.py`

```
usage: python -u convert_sn.py
```

**NOTE:** The paramater keys in pretrained model checkpoint must match the keys in backbone model **EXACTLY**.  You should load the correct pretrained model according to your segmentation architechure.


### Training

- The training strategies of baseline models and sn-based models on ADE20K are same as  [Scene Parsing on MIT ADE20K](https://github.com/CSAILVision/semantic-segmentation-pytorch).
- The training script with ResNet-50-sn backbone can be found here:  `./scripts/train.sh`

**NOTE:** The default architecture of this repo is `Encoder: resnet50_dilated8 ` ( resnetXX_dilatedYY: customized resnetXX with dilated convolutions, output feature map is 1/YY of input size, see [DeepLab](https://arxiv.org/abs/1606.00915.pdf) for more details ) and `Decoder: c1_bilinear_deepsup` ( 1 conv + bilinear upsample + deep supervision, see [PSPNet](https://arxiv.org/abs/1612.01105) for more details ).



Optional arguments (see full input arguments via `./train.py`):

```
  --arch_encoder         architecture of encode network
  --arch_decoder         architecture of decode network
  --weights_encoder      weights to finetune endoce network
  --weights_decoder      weights to finetune decode network
  --list_train           the list to load the training data 
  --root_dataset         the path of the dataset
  --batch_size_per_gpu   input batch size
  --start_epoch          epoch to start training. (continue from a checkpoint loaded via weights_encoder & weights_decoder)
  
```
**NOTE:**  In this repo, `--start_epoch` allows the training to resume from the checkpoint loaded from `--weights_encoder` and `--weights_decoder`, which is generated in the training process automatically. If you want to train from scratch, you need to assign `--start_epoch` as 1 and set `--weights_encoder` and `--weights_decoder`   to the blank value.


### Evaluation

- The evaluation script with ResNet-50-sn backbone can be found here : `./scripts/evaluate.sh`


Optional arguments (see full input arguments via `./eval.py`):

```
  --arch_encoder         architecture of encode network
  --arch_decoder         architecture of decode network
  --suffix               which snapshot to load
  --list_val             the list to load the validation data 
  --root_dataset         the path of the dataset
  --imgSize              list of input image sizes
```

`--imgSize` enables single-scale or multi-scale inference. When `--load_dir` is with the `int` type, the single-scale inference will be started up. When `--load_dir` is a `int list`,  the multi-scale test will be applied.


## Main Results

### Semantic Segmentation Results on ADE20K 

The experiment results are on the ADE20K validation set. MS test is short for multi-scale test. `sync BN` indicates the mutli-GPU synchronization batch normalization. More results and models will be released soon. 

|     Architecture      |  Norm   |   MS test  | Mean IoU |  Pixel Acc. |  Overall Score  | Download |
| :---:         |  :---:  |  :---:      |  :---:  |  :---:  |  :---:  |  :---:  |  
| ResNet50_dilated8 + c1_bilinear_deepsup| sync BN | no | 36.43 | 77.30 | 56.87 | [encoder](https://drive.google.com/file/d/1T0IAGpM1qIuT_74VGfuHyQ4QzYU3j55C/view?usp=sharing)  [decoder](https://drive.google.com/file/d/1fvrmSDQb58WHbUu-Ev15kidcaf7VwaFr/view?usp=sharing)  |
| ResNet50_dilated8 + c1_bilinear_deepsup| GN      | no | 35.66 | 77.24 | 56.45 | [encoder](https://drive.google.com/file/d/1YoXrwvfYzsHQ4P3IyVF2iThWzQtaTbGR/view?usp=sharing)  [decoder](https://drive.google.com/file/d/1HbuyhIiS3fPvBnHYG5xFRwj5Gpv5ULzT/view?usp=sharing)
| ResNet50_dilated8 + c1_bilinear_deepsup| SN-(8,2)| no | 38.72 | 78.90 | 58.82 | [encoder](https://drive.google.com/file/d/1Dn15_QTjdzX1pK3nvXHnHy94V7ffcKjL/view?usp=sharing)   [decoder](https://drive.google.com/file/d/1wS0lV9hWIBwWQ-Bhvdc1IRFyw3O_Fegx/view?usp=sharing) |
|||||
| ResNet50_dilated8 + c1_bilinear_deepsup| sync BN | yes | 37.69 | 78.29 | 57.99 | -- |
| ResNet50_dilated8 + c1_bilinear_deepsup| GN      | yes | 36.32 | 77.77 | 57.05 | -- |
| ResNet50_dilated8 + c1_bilinear_deepsup| SN-(8,2)| yes | 39.21 | 79.20 | 59.21 | -- |


**NOTE:** For all settings in this repo, we employ ResNet as the backbone network, using the original 7×7 kernel size in the first convolution layer. This is different from the [MIT framework](https://github.com/CSAILVision/semantic-segmentation-pytorch) , which adopts 3 convolution layers with the kernel size 3×3 at the bottom of the network. See  `./models/resnet_v1_sn.py` for the details.

