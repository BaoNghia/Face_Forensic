import PIL, cv2
import torch
import numpy as np
from torchvision import transforms
import albumentations as albu

# from imgaug import augmenters as iaa
# import imgaug as ia


train_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize(256),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(hue=.05, saturation=.05),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]
)

val_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize(256),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]
)

test_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize(256),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]
)
