import torch
from torch import nn
import torch.nn.functional as F
# from torchvision import models
from lib.syncbn import SwitchNorm2d
from models import resnet_v1_sn as models
from models import switchable_norm as sn 

# from utils import init_weights


class ResNet_Seg_SN(nn.Module):
    def __init__(self, layers=50, bins=(1, 2, 3, 6), dropout=0.1, classes=2, zoom_factor=8, use_softmax=True, use_aux=True, pretrained=False, syncbn=True):
        super(ResNet_Seg_SN, self).__init__()
        assert layers in [50, 101, 152]
        assert 2048 % len(bins) == 0
        assert classes > 1
        assert zoom_factor in [1, 2, 4, 8]
        self.zoom_factor = zoom_factor
        self.use_softmax = use_softmax
        self.use_aux = use_aux

#        if syncbn:
#            from lib.syncbn import SynchronizedBatchNorm2d as BatchNorm
#        else:
#            from torch.nn import BatchNorm2d as BatchNorm
#        models.BatchNorm = BatchNorm

        if layers == 50:
            resnet_sn = models.resnetv1sn50()
        elif layers == 101:
            resnet_sn = models.resnetv1sn101()
        else:
            resnet_sn = models.resnetv1sn152()

        self.layer0 = nn.Sequential(resnet_sn.conv1, resnet_sn.sn1, resnet_sn.relu, resnet_sn.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = resnet_sn.layer1, resnet_sn.layer2, resnet_sn.layer3, resnet_sn.layer4

        for n, m in self.layer3.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.layer4.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)

        fea_dim = 2048
        self.cls = nn.Sequential(
                nn.Conv2d(fea_dim, 512, kernel_size=3, padding=1, bias=False),
                #SwitchNorm2d(512, using_bn=True),
                sn.SwitchNorm2d(512, using_bn=True),
                nn.ReLU(inplace=True), 
                nn.Dropout2d(p=dropout), 
                nn.Conv2d(512, classes, kernel_size=1)
        )
        if use_aux:
            self.aux = nn.Sequential(
                nn.Conv2d(1024, 256, kernel_size=3, padding=1, bias=False),
                #SwitchNorm2d(256, using_bn=True),
                sn.SwitchNorm2d(256, using_bn=True),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=dropout),
                nn.Conv2d(256, classes, kernel_size=1)
            )
            # init_weights(self.aux)
        # comment to use default initialization
        # init_weights(self.ppm)
        # init_weights(self.cls)

    def forward(self, x):
        x_size = x.size()
        assert (x_size[2]-1) % 8 == 0 and (x_size[3]-1) % 8 == 0
        h = int((x_size[2] - 1) / 8 * self.zoom_factor + 1)
        w = int((x_size[3] - 1) / 8 * self.zoom_factor + 1)

        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        if self.training and self.use_aux:
            aux = self.aux(x)
            if self.zoom_factor != 1:
                aux = F.upsample(aux, size=(h, w), mode='bilinear', align_corners=True)
        x = self.layer4(x)
        x = self.cls(x)
        if self.zoom_factor != 1:
            x = F.upsample(x, size=(h, w), mode='bilinear', align_corners=True)

        if self.use_softmax:
            x = F.log_softmax(x, dim=1)
            if self.use_aux:
                aux = F.log_softmax(aux, dim=1)
                return x, aux
            else:
                return x
        else:
            if self.use_aux:
                return x, aux
            else:
                return x


if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    sim_data = torch.autograd.Variable(torch.rand(2, 3, 473, 473)).cuda(async=True)
    model = ResNet_Seg_SN(layers=50, bins=(1, 2, 3, 6), dropout=0.1, classes=21, zoom_factor=1, use_ppm=True, use_softmax=True, use_aux=True, pretrained=False, syncbn=True).cuda()
    print(model)
    output, _ = model(sim_data)
    print('ResNet_Seg_SN', output.size())
