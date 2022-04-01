import timm
import torch
import torch.nn as nn
from pprint import pprint
from torchsummary import summary



def load_backbone(name, pretrained = True):
    model_names = timm.list_models(pretrained=True)
    model_names = [name for name in model_names if 'efficientnetv2' in name]
    assert (name in model_names), f"name must be in {model_names}"

    print(f'Loading: {name}. Using pretrained \'imagenet\': {pretrained}')
    model = timm.create_model(name, pretrained=pretrained)
    for param in model.parameters():
        param.requires_grad = True
    return model


class EfficientnetV2_transfer(nn.Module):
    def __init__(self, model_name, num_classes, pretrained, **kwargs):
        super(EfficientnetV2_transfer, self).__init__()
        self.model = load_backbone(model_name, pretrained)
        self.backbone = nn.Sequential(*list(self.model.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.lastconv_output_channels = self.model.classifier.in_features
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
    model_names = timm.list_models(pretrained=True)
    # model = timm.create_model('tf_efficientnetv2_m', pretrained=True)
    model = EfficientnetV2_transfer('tf_efficientnetv2_m', num_classes=6, pretrained=True)

    features, x4, logits = model(torch.rand(5,3,256,256))
    print(features.shape, x4.shape, logits.shape)