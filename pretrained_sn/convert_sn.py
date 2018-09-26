import torch
import numpy as np


ckpt = torch.load('resnet-50-sn-ImageNet-pretrained.pth', map_location='cpu')
mapping = {}
for sd in ckpt['state_dict'].keys():
    print(sd)
    mapping[sd[7:]] = sd



state_dict = {}
for name in mapping:   
    state_dict[name] = ckpt['state_dict'][mapping[name]]
    print('%s\t\t%s'%(name,mapping[name]))


torch.save({'state_dict':state_dict},'resnet-50-sn-segmentation.pth')



