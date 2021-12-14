from torchvision import datasets, transforms
import torch

def cifar10_dataloader(cfg):
    # setup data loader
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    batch_size = cfg.get("data").get("batch_size")
    kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
    trainset = datasets.CIFAR10(root='./data/cifar10', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, **kwargs)
    testset = datasets.CIFAR10(root='./data/cifar10', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, **kwargs)
    return train_loader, test_loader


def cifar100_dataloader(cfg):
    # setup data loader
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    batch_size = cfg.get("data").get("batch_size")
    kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
    trainset = datasets.CIFAR10(root='./data/cifar100', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, **kwargs)
    testset = datasets.CIFAR10(root='./data/cifar100', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, **kwargs)
    return train_loader, test_loader