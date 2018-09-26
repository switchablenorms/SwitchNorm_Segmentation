#!/bin/sh
exp=res50_dilated8_c1_bilinear_deepsup_sn-8-2
EXP_DIR=exp/ADE20K/$exp
mkdir -p ${EXP_DIR}/model
now=$(date +"%Y%m%d_%H%M%S")
cp scripts/train.sh train.py ${EXP_DIR}


list_train={repo_root}/data/train.odgt
list_val={repo_root}/data/train.odgt



python3 -u train.py \
  --num_gpus $nodeGPU \
  --batch_size_per_gpu 2 \
  --root_dataset {your_path}/data/ \
  --list_train $list_train \
  --list_val $list_val \
  --arch_encoder resnet50_dilated8 \
  --arch_decoder c1_bilinear_deepsup \
  --lr_encoder 2e-2 \
  --lr_decoder 2e-2 \
  --padding_constant 8 \
  --segm_downsampling_rate 8 \
  --imgSize 300 375 450 525 575\
  --num_class 150 \
  --num_epoch 20 \
  --start_epoch 1 \
  --ckp ${EXP_DIR}/model \
  & \



