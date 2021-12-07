from torchvision import models
import torch.nn as nn
import sys
import os
import torch

def load_wideresnet(name, num_class = 2, pretrained = True):
    if not name in ['wide_resnet50_2', 'wide_resnet101_2']:
        raise ValueError("name must be in {'wide_resnet50_2', 'wide_resnet101_2'}")
        sys.exit()
    print(f'Loading: {name}. Using pretrained: {pretrained}')
    model = getattr(models, name)(pretrained=pretrained)
    for param in model.parameters():
        param.requires_grad = True
        
    classifier = nn.Sequential(
        nn.Linear(model.fc.in_features, 512, bias=True),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Linear(512, 256, bias=True),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Linear(256, num_class, bias=True),
    )
    model.fc = classifier
    return model

class WideResNet_transfer(nn.Module):
    def __init__(self, model_name, num_class, pretrained, **kwargs):
        super(WideResNet_transfer, self).__init__()
        self.model = load_wideresnet(model_name, num_class, pretrained)

    def forward(self, x):
        return self.model(x)
