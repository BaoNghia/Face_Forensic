from torchvision import models, transforms
from torchsummary import summary
import torch.nn as nn
import sys
import os
import torch

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    
    def forward(self, data):
        return data

class MaxL2Pool(nn.Module):
    def __init__(self):
        super(MaxL2Pool, self).__init__()
        self.conv = nn.Conv2d(in_channels = 512, 
                            out_channels = 512,
                            kernel_size = (3,3),
                            stride = 2,
                            padding = 1)
        self.conv1x1 = nn.Conv2d(in_channels = 512,
                            out_channels = 512, 
                            kernel_size = (1,1),
                            stride = 1,
                            padding = 0)
        self.max_pool = nn.AdaptiveMaxPool2d((1,1))
    
    def forward(self, data):
        data = self.conv(data)
        data = self.conv1x1(data)
        batch_size = data.shape[0]
        norm = torch.norm(data, dim = 1, keepdim=True)
        max_norm, _ = torch.max(norm.view(batch_size, -1), dim = 1, keepdim=False)
        max_norm = max_norm.view(batch_size,1,1,1)
        norm = norm / max_norm
        norm[norm<1] = 0
        data_ = data*norm
        return self.max_pool(data_)

class Pool(nn.Module):
    def __init__(self):
        super(Pool, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.max_pool = MaxL2Pool()
    
    def forward(self, data):
        batch_size = data.shape[0]
        max_data = self.max_pool(data)
        avg_data = self.avg_pool(data)
        # print(max_data.shape)
        # print(avg_data.shape)
        max_data = max_data.view(batch_size, -1)
        avg_data = avg_data.view(batch_size, -1)

        return torch.cat([max_data, avg_data], dim = 1)

def load_resnet(name, num_class = 2, pretrained = True):
    if not name in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']:
        raise ValueError("name must be in {'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'}")
        sys.exit()
    print(f'Loading: {name}. Using pretrained \'imagenet\': {pretrained}')
    model = getattr(models, name)(pretrained=pretrained)
    for param in model.parameters():
        param.requires_grad = True
        
    fc_layer = nn.Sequential(
        nn.Linear(model.fc.in_features, 256, bias=True),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Linear(256, num_class, bias=True),
    )
    model.fc = fc_layer
    return model

class ResNet_transfer(nn.Module):
    def __init__(self, model_name, num_classes, pretrained,**kwargs):
        super(ResNet_transfer, self).__init__()
        self.model = load_resnet(model_name, num_classes, pretrained)

    def forward(self, data):
        logits = self.model(data)
        return None, None, logits
    
if __name__ == '__main__':

    model = ResNet_transfer('resnet50', num_classes=2, pretrained=True)
    summary(model, (3,256,256))
