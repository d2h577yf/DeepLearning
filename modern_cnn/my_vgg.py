import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler

# /root/autodl-tmp/datasets/cifar-100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_cifar100(batch_size):
    data_root = '/root/autodl-tmp/datasets/cifar-100'
    
    transform_train = transforms.Compose([
        transforms.Resize((227,227)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize((227,227)),
        transforms.ToTensor(),
        transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
    ])
    
    train_set = torchvision.datasets.CIFAR100(
            root=data_root,
            train=True,
            download=True,
            transform=transform_train
    )
    
    test_set = torchvision.datasets.CIFAR100(
            root=data_root,
            train=False,
            download=True,
            transform=transform_test
    )
    
    train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=6
    )
    
    test_loader = torch.utils.data.DataLoader(
            test_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=6
    )
    
    return train_loader, test_loader

def build_feature():
    pass

def main():
    batch_size = 128
    train_loader,test_loader = load_cifar100(batch_size)
    print(len(train_loader))
    print(len(test_loader))

if __name__ == '__main__':
    main()