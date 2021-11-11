from torchvision import models
import torch.nn as nn
import sys


def load_densenet(name, num_class = 3):
    if not name in ['densenet121', 'densenet161', 'densenet169', 'densenet201']:
        raise ValueError("name must be in {'densenet121', 'densenet161', 'densenet169', 'densenet201'}")
        sys.exit()
    print('load {}'.format(name))
    model = getattr(models, name)(pretrained=False)
    for param in model.parameters():
        param.requires_grad = False
    fc_layer = nn.Sequential()
    # fc_layer.add_module('fc_1', nn.Linear(model.classifier.in_features, model.classifier.in_features, bias = True))
    # fc_layer.add_module('fc_1_act', nn.Sigmoid())
    fc_layer.add_module('fc_2', nn.Linear(model.classifier.in_features, num_class, bias = True))
    model.fc = fc_layer
    return model

class DenseNet_transfer(nn.Module):
    def __init__(self, name, num_class, **kwargs):
        super(DenseNet_transfer, self).__init__()
        self.model = load_densenet(name, num_class)

    def forward(self, data):
        return self.model(data)