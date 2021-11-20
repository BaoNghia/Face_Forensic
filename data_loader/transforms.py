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
        transforms.RandomResizedCrop(320),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(hue=.05, saturation=.05),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]
)

val_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.CenterCrop(320),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]
)

test_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize(320),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]
)


#Defince augmenter
# class ImgAugTransform:
#   def __init__(self):
#     self.aug = iaa.Sequential([
#         iaa.Scale((256, 256)),
#         iaa.Sometimes(0.25, iaa.GaussianBlur(sigma=(0, 3.0))),
#         iaa.Fliplr(0.5),
#         iaa.Affine(rotate=(-90, 90), mode='symmetric'),
#         iaa.Sometimes(0.25,
#                       iaa.OneOf([iaa.Dropout(p=(0, 0.1)),
#                                  iaa.CoarseDropout(0.1, size_percent=0.1)])),
#         iaa.AddToHueAndSaturation(value=(-10, 10), per_channel=True)
#     ])
      
#   def __call__(self, img):
#     img = np.array(img)
#     return self.aug.augment_image(img)


# train_transform = transforms.Compose([
#     ImgAugTransform(),
#     lambda x: PIL.Image.fromarray(x),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
# ])
