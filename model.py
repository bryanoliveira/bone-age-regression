#!/usr/bin/env python
# coding: utf-8

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class Model(nn.Module):

    def __init__(self, features):
        super(Model, self).__init__()
        
        self.features = features

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            Swish(),
            nn.Dropout(),
            nn.Linear(512, 512),
            Swish(),
            nn.Linear(512, 1),
        )
        
        self.loss = nn.MSELoss()

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.freeze_bn()
        self.training = False

    def forward(self, inputs):

        if self.training:
            x, y = inputs
        else:
            x = inputs

        y1 = self.features(x)
        y1 = self.avgpool(y1)
        y1 = y1.view(y1.size(0), -1)
        y1 = self.classifier(y1)
        
        if self.training:
            return self.loss(y1, y)
        else:
            return y1

   
    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()
                
def make_layers(cfg, batch_norm=True):
    layers = []
    in_channels = 1
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), Swish()]
            else:
                layers += [conv2d, Swish()]
            in_channels = v
    return nn.Sequential(*layers)
    
cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M', 512, 512],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M', 512, 512, 'M', 512, 512, 'M', 512, 512],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M', 512, 512, 512],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512],
}

if __name__ == '__main__':
    # test model
    model = Model(make_layers(cfg['B']))
    model.cuda()
    print(model)