from torchvision import models, transforms
import torch.nn as nn
import sys
from models.cnn_spp.spp import SPPLayer
def load_vggnet(name, num_class = 2,scale=[1]):
    if not name in ['vgg16']:
        raise ValueError("name must be in {'vgg16'}")
        sys.exit()
    model = getattr(models, name)(pretrained=True)
    for index,param in enumerate(model.parameters()):
        if index not in [1,2,3]:
            param.requires_grad = False
    print(model)
    fc_layer = nn.Sequential()
    fc_layer.add_module('fc_1', nn.Linear(25088, 4096, bias = True))
    fc_layer.add_module('fc_1_act', nn.ReLU())
    fc_layer.add_module('dropout_1', nn.Dropout(0.5))
    fc_layer.add_module('fc_2', nn.Linear(4096, 2048, bias = True))
    fc_layer.add_module('fc_2_act', nn.ReLU())
    fc_layer.add_module('dropout_2', nn.Dropout(0.5))
    fc_layer.add_module('fc_3', nn.Linear(2048, num_class, bias = True))

    model.classifier = fc_layer
    return model

class VGG_transfer(nn.Module):
    def __init__(self, name, num_class, **kwargs):
        super(VGG_transfer, self).__init__()
        self.model = load_vggnet(name, num_class)

    def forward(self, data):
        return self.model(data)