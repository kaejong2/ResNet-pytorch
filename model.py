import os
import numpy as np

import torch
import torch.nn as nn

class Layer(nn.Module):
    def __init__(self, in_features, out_features, kernel_size, stride, padding, norm, act):
        super(Layer, self).__init__()

        block = []
        block += [nn.Conv2d(in_features, out_features, kernel_size=kernel_size, stride=stride, padding=padding)]

        if not norm is False:
            if norm == "Batch":
                block += [nn.BatchNorm2d(out_features)]
            elif norm == "Instance":
                block += [nn.InstanceNorm2d(out_features)]
        if not act is False:
            if act == "ReLU":
                block += [nn.ReLU(inplace=True)]
            elif act == "LeakyReLU":
                block += [nn.LeakyReLU(0.2, inplace=True)]
            elif act == "Tanh":
                block += [nn.Tanh()]
            elif act == "Sigmoid":
                block += [nn.Sigmoid()]
        
        self.layer = nn.Sequential(*block)

    def forward(self, x):
        return self.layer(x)
    


class Resblock(nn.Module):
    def __init__(self, in_features, out_features, kernel_size, stride, padding, norm, act, expension):
        super(Resblock, self).__init__()
        block = []
        self.downsampling = expension
        if expension == 0 and out_features != 64:
            block += [Layer(in_features, out_features, kernel_size, stride=2, padding=padding, norm=norm, act=act)]
            in_features=out_features
        else: block += [Layer(in_features, out_features, kernel_size, stride, padding, norm, act)]

        block += [Layer(in_features, out_features, kernel_size, stride, padding, norm, act)]

        self.resblock = nn.Sequential(*block)
    
    def forward(self, x):
        if self.downsampling ==0:
            return self.resblock(x)
        else: return x + self.resblock(x)

# resblock = [3, 4, 6, 3]
# for b in resblock:
#     for j in range(b):
#         print(j)

class ResNet(nn.Module):
    def __init__(self, in_channels, num_classes, resblock):
        super(ResNet, self).__init__()

        self.layer1 = Layer(in_features=in_channels, out_features=64, kernel_size=7, stride=2, padding=3, norm='Batch', act="ReLU")
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        in_feature = 64
        out_feature = 64
        res = []
        for block in resblock:
            for i in range(block):
                res += [Resblock(in_features=in_feature, out_features=out_feature, kernel_size=3, stride=1, padding=1, norm='Batch', act='ReLU', expension=i)]
                in_feature = out_feature
            out_feature = out_feature*2
        self.layer2 = nn.Sequential(*res)
        self.final_pool = nn.AdaptiveAvgPool2d((1,1))
        self.linear = nn.Linear(512, num_classes)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.maxpool(x)
        x = self.layer2(x)
        x = self.final_pool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)

        return x