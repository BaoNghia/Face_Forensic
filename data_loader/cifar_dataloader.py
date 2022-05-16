from torchvision import datasets, transforms
import torch

def cifar10_dataset(cfg):
    # setup data loader
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    trainset = datasets.CIFAR10(root='./data/cifar10', train=True, download=True, transform=transform_train)
    testset = datasets.CIFAR10(root='./data/cifar10', train=False, download=True, transform=transform_test)
    return trainset, testset


def cifar100_dataset(cfg):
    # setup data loader
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    trainset = datasets.CIFAR10(root='./data/cifar100', train=True, download=True, transform=transform_train)
    testset = datasets.CIFAR10(root='./data/cifar100', train=False, download=True, transform=transform_test)
    return trainset, testset