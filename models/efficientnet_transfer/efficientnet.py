import sys
import torch
import torch.nn as nn
from torchvision import models
from torchsummary import summary


def load_backbone(name, pretrained = True):
    model_name = ['efficientnet_b0','efficientnet_b1', 'efficientnet_b2', \
                'efficientnet_b3','efficientnet_b4','efficientnet_b5', \
                'efficientnet_b6','efficientnet_b7']
    if not name in model_name:
        raise ValueError(f"name must be in {model_name}")
        sys.exit()
    
    print(f'Loading: {name}. Using pretrained \'imagenet\': {pretrained}')
    model = getattr(models, name)(pretrained=pretrained)
    for param in model.parameters():
        param.requires_grad = True

    return model

class Efficientnet_transfer(nn.Module):
    def __init__(self, model_name, num_classes, pretrained, **kwargs):
        super(Efficientnet_transfer, self).__init__()
        self.model = load_backbone(model_name, pretrained)
        self.backbone = nn.Sequential(*list(self.model.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.lastconv_output_channels = self.model.classifier[1].in_features
        self.fcs = nn.Sequential(
            nn.Linear(self.lastconv_output_channels, 512, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, 256, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes, bias=True)
        )

    def forward(self, x):
        features = self.backbone(x)
        x4 = self.avgpool(features)
        x4 = torch.flatten(x4, 1)
        logits = self.fcs(x4)
        return features, x4, logits

if __name__ == '__main__':
    model = Efficientnet_transfer('efficientnet_b5', num_classes=6, pretrained=True)
    print(model)

    features, x4, logits = model(torch.rand(5,3,256,256))
    print(features.shape, x4.shape, logits.shape)