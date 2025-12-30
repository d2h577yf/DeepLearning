import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import tomllib as toml
import time
import train_predict as tp

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
            pin_memory = True,
            num_workers=6
    )
    
    test_loader = torch.utils.data.DataLoader(
            test_set,
            batch_size=batch_size,
            shuffle=False,
            pin_memory = True,
            num_workers=6
    )
    
    return train_loader, test_loader


class VGG_model(nn.Module):
    def __init__(self, toml_path: str):
        super().__init__()
        self.toml_path = toml_path
        self.config = self._load_config()
        self.features, self.classifier = self._build_model()
    
    def _load_config(self):
        with open(self.toml_path, 'rb') as f:
            config = toml.load(f)
        return config
    
    def _build_model(self):
        features = nn.Sequential()
        classifier = nn.Sequential()
        
        in_channels = self.config['model']['input_channel']
        fc_input_size = None
        
        for i in range(1, 14):
            layer_key = f'l{i}'
            if layer_key in self.config:
                layer_config = self.config[layer_key]
                
                if layer_config['type'] == 'conv':
                    conv = nn.Conv2d(
                            in_channels,
                            layer_config['ch'],
                            kernel_size=3,
                            stride=1,
                            padding=1
                    )
                    features.add_module(f'conv{i}', conv)
                    
                    if layer_config.get('bn', True):
                        bn = nn.BatchNorm2d(layer_config['ch'])
                        features.add_module(f'bn{i}', bn)
                    
                    features.add_module(f'relu{i}', nn.ReLU())
                    
                    if layer_config.get('pool', False):
                        pool = nn.MaxPool2d(kernel_size=2, stride=2)
                        features.add_module(f'pool{i//2}', pool)  # 简化命名
                    
                    in_channels = layer_config['ch']
        
        features.add_module('adaptive_pool', nn.AdaptiveAvgPool2d((7, 7)))
        
        classifier.add_module('flatten', nn.Flatten())
        
        for i in range(14, 17):
            layer_key = f'l{i}'
            if layer_key in self.config:
                layer_config = self.config[layer_key]
                
                if layer_config['type'] == 'fc':
                    classifier.add_module(f'dropout{i-13}',
                                          nn.Dropout(0.5))
                    linear = nn.Linear(*layer_config['ch'])
                    classifier.add_module(f'fc{i-13}', linear)
                    classifier.add_module(f'fc_relu{i-13}', nn.ReLU())
                
                elif layer_config['type'] == 'out':
                    linear = nn.Linear(*layer_config['ch'])
                    classifier.add_module('output', linear)
        
        return features, classifier
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x
    
def main():
    batch_size = 256
    lr = 0.1
    num_epochs = 100
    w_d = 5e-4
    mt = 0.9
    
    train_loader, test_loader = load_cifar100(batch_size)
    
    model = VGG_model('../Config/vgg_config.toml')
    
    def init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    
    model.apply(init_weights)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            weight_decay=w_d,
            momentum=mt,
            nesterov=True
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_epochs,
            eta_min=1e-6
    )
    
    
    start_time = time.time()
    
    train_losses, test_accuracies, best_accuracy = tp.train_model(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            loss_fn=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            epochs=num_epochs,
            device='cuda',
            save_path='best_model.pth',
            log_file='train_log.txt',
    )
    
    end_time = time.time()

    
    print("\n使用最佳模型进行测试...")
    predictions, accuracy, total, correct, labels = tp.predict_model(
            model = model,
            data_loader = test_loader,
            model_path = 'best_model.pth',
            log_file = 'test_log.txt'
    )
    
    print(f"\n训练用时: {end_time - start_time:.2f}秒")
    print(f"最终测试准确率: {accuracy:.2f}%")
    print(f"正确数/总数: {correct}/{total}")

if __name__ == '__main__':
    main()