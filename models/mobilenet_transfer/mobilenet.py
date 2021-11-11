from torchvision import models, transforms
import torch.nn as nn
import sys
def load_mobilenet(name, num_class = 2):
    if not name in ['mobilenet', 'mobilenet_v2']:
        raise ValueError("name must be in {'mobilenet', 'mobilenet_v2'}")
        sys.exit()
    model = getattr(models, name)(pretrained=True)
    print(model)
    for param in model.parameters():
        param.requires_grad = False
    fc_layer = nn.Sequential()
    fc_layer.add_module('fc_1', nn.Linear(1280, 512, bias = True))
    fc_layer.add_module('fc_1_act', nn.Sigmoid())
    fc_layer.add_module('fc_2', nn.Linear(512, num_class, bias = True))
    model.classifier = fc_layer
    return model

class Mobilenet_transfer(nn.Module):
    def __init__(self, name, num_class, **kwargs):
        super(Mobilenet_transfer, self).__init__()
        self.model = load_mobilenet(name, num_class)

    def forward(self, data):
        return self.model(data)