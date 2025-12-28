# Library import
import time
import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn
import train_predict as tp


def load_cifar10(batch_size):
    data_root = '/root/autodl-tmp/datasets/cifar-10'
    
    transform_train = transforms.Compose([
        # transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
    ])
    
    transform_test = transforms.Compose([
        # transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
    ])
    
    train_set = torchvision.datasets.CIFAR10(
            root=data_root,
            train=True,
            download=True,
            transform=transform_train
    )
    
    test_set = torchvision.datasets.CIFAR10(
            root=data_root,
            train=False,
            download=True,
            transform=transform_test
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


class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
                nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
                nn.ReLU(inplace=True),
                nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2),
                nn.MaxPool2d(kernel_size=3, stride=2),
                
                nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
                nn.ReLU(inplace=True),
                nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2),
                nn.MaxPool2d(kernel_size=3, stride=2),
                
                nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
        )
        
        self.classifier = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Linear(256 * 6 * 6, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, 10),
        )
    
    def forward(self, x: torch.Tensor):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class AlexNetCIFAR(nn.Module):
    """专门为CIFAR-10修改的AlexNet"""
    
    def __init__(self, num_classes = 10):
        super().__init__()
        self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size = 5, stride = 1, padding = 2),  # 32×32
                nn.ReLU(inplace = True),
                nn.LocalResponseNorm(size = 5),
                nn.MaxPool2d(kernel_size = 3, stride = 2),  # 32→15
                
                nn.Conv2d(64, 128, kernel_size = 5, stride = 1, padding = 2),  # 15×15
                nn.ReLU(inplace = True),
                nn.LocalResponseNorm(size = 5),
                nn.MaxPool2d(kernel_size = 3, stride = 2),  # 15→7
                
                nn.Conv2d(128, 256, kernel_size = 3, stride = 1, padding = 1),  # 7×7
                nn.ReLU(inplace = True),
                nn.Conv2d(256, 256, kernel_size = 3, stride = 1, padding = 1),
                nn.ReLU(inplace = True),
                nn.Conv2d(256, 128, kernel_size = 3, stride = 1, padding = 1),
                nn.ReLU(inplace = True),
                nn.MaxPool2d(kernel_size = 3, stride = 2),  # 7→3
        )
        
        self.classifier = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(1152, 1024),  # 减小全连接层
                nn.ReLU(inplace = True),
                nn.Dropout(0.5),
                nn.Linear(1024, 512),
                nn.ReLU(inplace = True),
                nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0.0, std=0.01)
        
        if isinstance(m, nn.Conv2d) and (m.out_channels in [256, 384]):
            nn.init.constant_(m.bias, 1.0)
        elif isinstance(m, nn.Linear) and m.out_features == 4096:
            nn.init.constant_(m.bias, 1.0)
        else:
            nn.init.constant_(m.bias, 0.0)


def main():
    batch_size = 128
    train_loader, test_loader, classes = load_cifar10(batch_size)
    
    # model = AlexNet()
    model = AlexNetCIFAR()
    
    model.apply(init_weights)
    
    start_time = time.time()
    
    train_losses, test_accuracies, best_accuracy = tp.train_model(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            epochs=60,
            lr=0.01,
            w_d=0.0005
    )
    
    end_time = time.time()
    
    final_accuracy, total, correct = tp.test_model(model, test_loader)
    print(f"\n训练用时: {end_time - start_time:.2f}秒")
    print(f"最终测试准确率: {final_accuracy:.2f}%")
    print(f"正确数/总数: {correct}/{total}")


if __name__ == '__main__':
    main()