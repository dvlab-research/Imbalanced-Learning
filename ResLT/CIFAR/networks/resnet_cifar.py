"""

Properly implemented ResNet-s for CIFAR10 as described in paper [1].
The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.
Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:
name      | layers | params
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4m
which this implementation indeed has.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
import math

BN=nn.BatchNorm2d

__all__ = [
    "ResNet",
    "resnet20",
    "resnet32",
    "resnet44",
    "resnet56",
    "resnet110",
    "resnet1202",
]


def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, groups=1, option="A", arrange=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, groups=groups
        )
        self.bn1 = BN(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
        )
        self.bn2 = BN(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == "A":
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(
                    lambda x: F.pad(
                        x[:, :, ::2, ::2],
                        (0, 0, 0, 0, planes // 4, planes // 4),
                        "constant",
                        0,
                    )
                )
            elif option == "B":
                self.shortcut = nn.Sequential(
                    nn.Conv2d(
                        in_planes,
                        self.expansion * planes,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    BN(self.expansion * planes),
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet_Cifar(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, scale=1, groups=1, nc=[16, 32, 64], arrange=[2,1,1]):
        super(ResNet_Cifar, self).__init__()
        self.in_planes = nc[0] * scale
        self.arrange=arrange

        self.conv1 = nn.Conv2d(3, nc[0] * scale, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = BN(nc[0] * scale)
        self.layer1 = self._make_layer(block, nc[0] * scale , num_blocks[0], stride=1, groups=groups)
        self.layer2 = self._make_layer(block, nc[1] * scale , num_blocks[1], stride=2, groups=groups)
        self.layer3 = self._make_layer(block, nc[2] * scale , num_blocks[2], stride=2, groups=groups)
        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride, groups=1):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, groups=groups, arrange=self.arrange))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x, **kwargs):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        return out

class ResLTResNet32(nn.Module):
      def __init__(self, num_classes=10, scale=1):
          super(ResLTResNet32, self).__init__()
          nc=[16, 32, 64]
          nc=[c * scale for c in nc]
          self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
          self.linear = nn.Linear(nc[2], num_classes)
          self.model = ResNet_Cifar(BasicBlock, [5, 5, 5], num_classes=num_classes, nc=nc)

          # 1x1 conv can be replaced with more light-weight bn layer
          self.BNH = nn.BatchNorm2d(nc[2])
          self.BNM = nn.BatchNorm2d(nc[2])
          self.BNT = nn.BatchNorm2d(nc[2])

      def forward(self, x):
          out = self.model(x)
          head_fs, medium_fs, tail_fs=self.BNH(out), self.BNM(out), self.BNT(out) 

          fs = torch.cat((head_fs, medium_fs, tail_fs),dim=0)
          logits = self.linear(self.avgpool(fs).view(fs.size(0),-1))
          c = logits.size(0) // 3
          return  logits[:c,:], logits[c:c*2,:], logits[c*2:,:]
