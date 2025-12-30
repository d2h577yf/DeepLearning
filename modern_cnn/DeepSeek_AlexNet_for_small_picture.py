import time
import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn
import torch.optim as optim

from train_predict import train_model, predict_model, Logger


class CIFAR10Net(nn.Module):
    """基于配置文件的CIFAR-10专用网络"""
    def __init__(self, num_classes=10):
        super().__init__()
        
        # 按照配置文件构建
        self.features = nn.Sequential(
                # conv1: 5×5, 64, stride=1, padding=2
                nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2),
                nn.ReLU(inplace=True),
                
                # pool1: max, 3×3, stride=2
                nn.MaxPool2d(kernel_size=3, stride=2),
                
                # conv2: 5×5, 64, stride=1, padding=2
                nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
                nn.ReLU(inplace=True),
                
                # pool2: max, 3×3, stride=2
                nn.MaxPool2d(kernel_size=3, stride=2),
        )
        
        # 局部连接层
        self.local_layers = nn.Sequential(
                # local3: 3×3, 32, stride=1, padding=1
                nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                
                # local4: 3×3, 32, stride=1, padding=1
                nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
        )
        
        # 全连接层
        self.classifier = nn.Sequential(
                nn.Linear(7 * 7 * 32, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.local_layers(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def load_cifar10(batch_size=128, data_root='/root/autodl-tmp/datasets/cifar-10'):
    """加载CIFAR-10数据集"""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2023, 0.1994, 0.2010]
        )
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2023, 0.1994, 0.2010]
        )
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
            num_workers=2,
            pin_memory=True
    )
    
    test_loader = torch.utils.data.DataLoader(
            test_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
    )
    
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')
    
    return train_loader, test_loader, classes


def init_cifar10_weights(model):
    """按照配置文件的初始化策略初始化权重"""
    def init_weights(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            if hasattr(m, 'init_scale'):
                nn.init.normal_(m.weight, mean=0.0, std=m.init_scale)
            else:
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
    
    model.features[0].init_scale = 0.0001  # conv1: initW=0.0001
    model.features[3].init_scale = 0.01    # conv2: initW=0.01
    model.local_layers[0].init_scale = 0.04  # local3: initW=0.04
    model.local_layers[2].init_scale = 0.04  # local4: initW=0.04
    if isinstance(model.classifier[0], nn.Linear):
        model.classifier[0].init_scale = 0.01  # fc10: initW=0.01
    
    model.apply(init_weights)
    return model


def main():
    """主函数"""
    batch_size = 128
    epochs = 100
    lr = 0.01
    weight_decay = 0.0005
    momentum = 0.9
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    model_save_path = f'best_model_cifar10_{timestamp}.pth'
    log_file = f'cifar10_training_{timestamp}.txt'
    
    print("加载CIFAR-10数据集...")
    train_loader, test_loader, classes = load_cifar10(batch_size)
    
    print("创建模型...")
    model = CIFAR10Net(num_classes=10)
    
    model = init_cifar10_weights(model)
    
    print("\n模型架构:")
    print(model)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n总参数量: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    
    test_input = torch.randn(4, 3, 32, 32)
    output = model(test_input)
    print(f"\n测试输入: {test_input.shape}")
    print(f"测试输出: {output.shape}")
    
    optimizer = optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay
    )
    
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    
    print(f"\n开始训练，共{epochs}个epoch...")
    print(f"日志文件: {log_file}")
    print(f"模型将保存到: {model_save_path}")
    
    start_time = time.time()
    
    train_losses, test_accuracies, best_accuracy = train_model(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            epochs=epochs,
            device='cuda',
            save_path=model_save_path,
            log_file=log_file
    )
    
    end_time = time.time()
    
    print("\n使用最佳模型进行测试...")
    predictions, accuracy, total, correct, labels = predict_model(
            model=model,
            data_loader=test_loader,
            device='cuda',
            model_path=model_save_path,
            log_file=f'cifar10_test_{timestamp}.txt'
    )
    
    training_time = end_time - start_time
    
    print("\n" + "=" * 60)
    print("训练完成！")
    print("=" * 60)
    print(f"训练用时: {training_time:.2f}秒 ({training_time/60:.2f}分钟)")
    print(f"训练过程中最佳准确率: {best_accuracy:.2f}%")
    print(f"最终测试准确率: {accuracy:.2f}%")
    print(f"正确数/总数: {correct}/{total}")
    
if __name__ == '__main__':
    main()
    