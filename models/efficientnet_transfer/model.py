from efficientnet_pytorch import EfficientNet
from torchvision import models
import torch.nn as nn
import torch
import sys

def unfreeze(model):
    for param in model.parameters():
        param.requires_grad = True
def load_efficientnet(name,num_class = 3):
    if not name in ['efficientnet-b0','efficientnet-b1','efficientnet-b2','efficientnet-b3','efficientnet-b4','efficientnet-b5','efficientnet-b6','efficientnet-b7']:
        # raise ValueError("name must be in {efficientnet-b1}")
        sys.exit()
    # print('load {}'.format(name))
    
    model = EfficientNet.from_pretrained(name, num_classes = num_class,advprop=False)
    # model = EfficientNet.from_name(name)
    for index,param in enumerate(model.parameters()):
        # if index != 300:
        param.requires_grad = False
    # model.fc = nn.Sequential(
    #                   nn.Linear(1280, 256), 
    #                   nn.ReLU(), 
    #                   nn.Dropout(0.4),
    #                   nn.Linear(256, num_class),                   
    #                   nn.LogSoftmax(dim=1))
    # fc_layer = nn.Sequential()
    # fc_layer.add_module('fc_1', nn.Linear(1280, 256, bias = True))
    # fc_layer.add_module('fc_1_act', nn.Sigmoid())
    # fc_layer.add_module('fc_2', nn.Linear(265, num_class, bias = True))
    # model.fc = fc_layer
    model._fc = nn.Linear(1280,num_class)
    # model.load_state_dict(torch.load('/data1/phuong/checkpoints/Efficientnet_transfer/1591695026_none/Checkpoint.pth'))
    # unfreeze(model)
    print(model)
    return model
class Efficientnet_transfer(nn.Module):
    def __init__(self, name, num_class, **kwargs):
        super(Efficientnet_transfer, self).__init__()
        self.model = load_efficientnet(name, num_class)

    def forward(self, data):
        return self.model(data)

