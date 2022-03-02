import sys
import torch
import torch.nn as nn
from torchvision import models
from torchsummary import summary


def load_backbone(name, pretrained = True):
    if not name in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']:
        raise ValueError("name must be in {'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'}")
        sys.exit()
        
    model = getattr(models, name)(pretrained=pretrained)
    for param in model.parameters():
        param.requires_grad = True

    return model

class Resnet_transfer(nn.Module):
    def __init__(self, model_name, num_classes, pretrained,**kwargs):
        super(Resnet_transfer, self).__init__()
        self.model = load_backbone(model_name, pretrained)
        self.backbone = nn.Sequential(*list(self.model.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fcs = nn.Sequential(
            nn.Linear(self.model.fc.in_features, 512, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, 256, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes, bias=True),
        )

    def forward(self, x):
        features = self.backbone(x)
        x4 = self.avgpool(features)
        x4 = torch.flatten(x4, 1)
        logits = self.fcs(x4)
        return features, x4, logits
    
if __name__ == '__main__':
    model = Resnet_transfer('resnet50', num_classes=2, pretrained=True)
    features, x4, logits = model(torch.rand(5,3,256,256))
    print(features.shape, x4.shape, logits.shape)

