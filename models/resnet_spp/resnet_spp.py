import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from torch.autograd import Variable
from models.resnet_spp.spp import SPPLayer

__all__ = ['ResNet', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']

def _weights_init(m):
    classname = m.__class__.__name__
    # print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='B'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, in_channel, scales, num_features):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(in_channel, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.spp = SPPLayer(scales)
        self.linear = nn.Linear(64*sum([x**2 for x in scales]), num_features)
        
        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        # Apply SPP
        out = self.spp(out)
        out = self.linear(out)
        return out
        
    def get_embedding(self, x):
        return self.forward(x)

class ClassificationNet(nn.Module):
    def __init__(self, embedding_net, num_features, num_classes):
        super(ClassificationNet, self).__init__()
        self.num_classes = num_classes
        self.embedding_net = embedding_net
        self.linear = nn.Linear(num_features, num_classes, bias=False)
        # self.sigmoid = nn.Sigmoid()
        self.apply(_weights_init)

    def forward(self, x):
        output = self.embedding_net(x)
        output = self.linear(output)
        scores = F.log_softmax(output, dim=-1)  
        # scores = self.sigmoid(output)
        return scores

    def get_embedding(self, x):
        return self.embedding_net(x)
        
    def _get_name(self):
        return 'ClassificationNet'
        
def resnet20(in_channel = 3, num_features = 64, scales = [1], num_classes = 3):
    embedding_net = ResNet(BasicBlock, [3, 3, 3], in_channel, scales, num_features)
    resnet = ClassificationNet(embedding_net, num_features, num_classes)
    return resnet


def resnet32(in_channel = 3, num_features = 64, scales = [1], num_classes = 3):
    embedding_net = ResNet(BasicBlock, [5, 5, 5], in_channel, scales, num_features)
    resnet = ClassificationNet(embedding_net, num_features, num_classes)
    return resnet
    
    
def resnet44(in_channel = 3, num_features = 64, scales = [1], num_classes = 3):
    embedding_net = ResNet(BasicBlock, [7, 7, 7], in_channel, scales, num_features)
    resnet = ClassificationNet(embedding_net, num_features, num_classes)
    return resnet


def resnet56(in_channel =3, num_features = 64, scales = [1], num_classes = 3):
    embedding_net = ResNet(BasicBlock, [9, 9, 9], in_channel, scales, num_features)
    resnet = ClassificationNet(embedding_net, num_features, num_classes)
    return resnet


def resnet110(in_channel, num_features = 64, scales = [1], num_classes = 3):
    embedding_net = ResNet(BasicBlock, [18, 18, 18], in_channel, scales, num_features)
    resnet = ClassificationNet(embedding_net, num_features, num_classes)
    return resnet


def resnet1202(in_channel, num_features = 64, scales = [1], num_classes = 3):
    embedding_net = ResNet(BasicBlock, [200, 200, 200], in_channel, scales, num_features)
    resnet = ClassificationNet(embedding_net, num_features, num_classes)
    return resnet

def load_resetnetSPP(name, num_class = 2):
    if name.startswith('resnet'):
        print(name)
        model = globals()[name]()
    # model = models.mnasnet1_0(pretrained=True)
    # print(model)
    return model

class RestnetSPP(nn.Module):
    def __init__(self, model_name, num_class, **kwargs):
        super(RestnetSPP, self).__init__()
        self.model = load_resetnetSPP(model_name, num_class)

    def forward(self, data):
        return self.model(data)

if __name__ == '__main__':
    model = load_resetnetSPP('resnet44')