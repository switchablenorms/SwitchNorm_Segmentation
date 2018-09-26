#!/bin/sh

exp=res50_dilated8_c1_bilinear_deepsup_sn-8-2
EXP_DIR=exp/ADE20K/$exp
now=$(date +"%Y%m%d_%H%M%S")


id=ignore_this_argument

list_val={repo_root}/data/validation.odgt


python3 -u eval.py \
   --id $id \
   --suffix _epoch_20.pth \
   --root_dataset {your_path}/data/ \
   --list_val $list_val \
   --arch_encoder resnet50_dilated8 \
   --arch_decoder c1_bilinear_deepsup \
   --num_val 2000 \
   --num_class 150 \
   --imgSize 300 400 500 600 \
   --imgMaxSize 1000 \
   --padding_constant 8 \
   --ckpt ${EXP_DIR}/model/ \
   --result ${EXP_DIR}/result/ \
   --visualize \
   &\

#--imgSize 300 400 500 600 for multi-scale inference
#--imgSize 450 for single-scale inference
