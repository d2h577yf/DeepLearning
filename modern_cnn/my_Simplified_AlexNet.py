# Library import
import time
import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn
from torch.utils import data
import train_predict as tp

def load_cifar10(batch_size):
    data_root = '/root/autodl-tmp/datasets/cifar-10'
    
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
    ])

    
    train_set = torchvision.datasets.CIFAR10(
            root=data_root,
            train=True,
            download=True,
            transform=transform
    )
    
    test_set = torchvision.datasets.CIFAR10(
            root=data_root,
            train=False,
            download=True,
            transform=transform
    )
    
    train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=5
    )
    test_loader = torch.utils.data.DataLoader(
            test_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=5
    )
    
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')
    
    return train_loader, test_loader, classes

class

def main():
    batch_size = 128
    train_loader,test_loader,classes = load_cifar10(batch_size)
    print(f"训练集批次数量: {len(train_loader)}")
    print(f"测试集批次数量: {len(test_loader)}")

if __name__ == '__main__':
    main()