from torchvision import models
import torch.nn as nn
import sys

def load_mnastnet(name,num_class = 3):
    print('load {}'.format(name))
    model = models.mnasnet1_0(pretrained=False)
    # for param in model.parameters():
    #     param.requires_grad = False
    
    model.classifier[1] = nn.Linear(1280,num_class)
    print(model)
    return model
class Mnastnet_transfer(nn.Module):
    def __init__(self, name, num_class, **kwargs):
        super(Mnastnet_transfer, self).__init__()
        self.model = load_mnastnet(name, num_class)

    def forward(self, data):
        return self.model(data)

