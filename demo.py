from models.denoise_Resnet.denoiseResnet import Denoise_Resnet
import torch


if __name__ == '__main__':
    x = torch.rand(5,3,256,256)
    model = Denoise_Resnet('resnet50', num_classes=2, pretrained=False)
    features, x4, logits = model(x)
    print(features.shape, x4.shape, logits.shape)