"""Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""


import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import autocast

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class NormedLinear(nn.Module):

    def __init__(self, in_features, out_features):
        super(NormedLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        out = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        return out

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
    
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 groups=1, base_width=64, is_last=False):
        super(Bottleneck, self).__init__()
        width = int(planes * (base_width / 64.)) * groups
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride,
                               groups=groups, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.is_last = is_last

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNext(nn.Module):

    def __init__(self, block, layers, num_experts, groups=1, width_per_group=64, dropout=None, num_classes=1000, use_norm=False, reduce_dimension=False, layer3_output_dim=None, layer4_output_dim=None, returns_feat=False, s=30, gamma=0.7):

        self.gamma = 0.7
        self.inplanes = 64
        self.num_experts = num_experts
        super(ResNext, self).__init__()

        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.inplanes = self.next_inplanes
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.inplanes = self.next_inplanes

        if layer3_output_dim is None:
            if reduce_dimension:
                layer3_output_dim = 192
            else:
                layer3_output_dim = 256

        if layer4_output_dim is None:
            if reduce_dimension:
                layer4_output_dim = 384
            else:
                layer4_output_dim = 512
        
        self.layer3s = nn.ModuleList([self._make_layer(block, layer3_output_dim, layers[2], stride=2) for _ in range(num_experts)])
        self.inplanes = self.next_inplanes
        self.layer4s = nn.ModuleList([self._make_layer(block, layer4_output_dim, layers[3], stride=2) for _ in range(num_experts)])
        self.inplanes = self.next_inplanes


        ##### ResLT implementation 
        def ResLTBlock():
            return nn.Sequential(
                       nn.Conv2d(self.next_inplanes, self.next_inplanes * 3, 1, bias=False),
                       nn.BatchNorm2d(self.next_inplanes * 3),
                       nn.ReLU(inplace=True)
                   ) 

        self.ResLTBlocks = nn.ModuleList([ResLTBlock() for _ in range(num_experts)])
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.use_dropout = True if dropout else False

        if self.use_dropout:
            print('Using dropout.')
            self.dropout = nn.Dropout(p=dropout)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        if use_norm:
            self.linears = nn.ModuleList([NormedLinear(layer4_output_dim * block.expansion, num_classes) for _ in range(num_experts)])
        else:
            self.linears = nn.ModuleList([nn.Linear(layer4_output_dim * block.expansion, num_classes) for _ in range(num_experts)])
            s = 1
        
        self.s = s

        self.returns_feat = returns_feat

    def _make_layer(self, block, planes, blocks, stride=1, is_last=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,
                            groups=self.groups, base_width=self.base_width))
        self.next_inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.next_inplanes, planes,
                                groups=self.groups, base_width=self.base_width,
                                is_last=(is_last and i == blocks-1)))

        return nn.Sequential(*layers)

    def _separate_part(self, x, ind):
        x = (self.layer3s[ind])(x)
        x = (self.layer4s[ind])(x)

        # ResLT
        if self.training:
           x = (self.ResLTBlocks[ind])(x)
           x = self.avgpool(x)
        else:
           x = self.avgpool(x)
           x = (self.ResLTBlocks[ind])(x)
        x = torch.flatten(x, 1)

        c = x.size(1) // 3 
        bt = x.size(0)
        x1, x2, x3 = x[:,:c], x[:,c:c*2], x[:,c*2:c*3]
        out = torch.cat((x1, x2, x3),dim=0) 

        if self.use_dropout:
            out = self.dropout(out)

        if self.training:
           y = self.linears[ind](out)
        else:
           weight = self.linears[ind].weight
           norm = torch.norm(weight, 2, 1, keepdim=True)
           weight = weight / torch.pow(norm, self.gamma)
           y = torch.mm(out, torch.t(weight)) 
  
        y = y * self.s
        return (y[:bt,:], y[bt:bt*2,:], y[bt*2:bt*3,:])


    def forward(self, x):
        with autocast():
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)

            outs = []
            for ind in range(self.num_experts):
                outs.append(self._separate_part(x, ind))
            
        return outs


def ResNeXt50Model(num_classes=1000, reduce_dimension=True, layer3_output_dim=None, layer4_output_dim=None, num_experts=3, gamma=0.7, **kwargs):
    return ResNext(Bottleneck, [3, 4, 6, 3], groups=32, width_per_group=4, dropout=None, num_classes=num_classes, reduce_dimension=reduce_dimension, layer3_output_dim=layer3_output_dim, layer4_output_dim=layer4_output_dim, num_experts=num_experts, **kwargs)

