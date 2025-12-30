import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import tomllib as toml
from dataclasses import dataclass, field
from typing import List, Union
import time
import train_predict as tp

@dataclass
class DataConfig:
    input_channels: int = 3
    num_classes: int = 100
    data_path: str = "/root/autodl-tmp/datasets/cifar-100"
    model_save : str = './best_model.pth'

@dataclass
class LogConfig:
    train_log_save_path: str = "./train_log.txt"
    test_log_save_path: str = "./test_log.txt"

@dataclass
class ModelConfig:
    conv_blocks: List[List[Union[int, str]]] = field(default_factory=list)
    fc_layers: List[List[Union[int, str, float]]] = field(default_factory=list)

@dataclass
class OptimizerConfig:
    name: str = "sgd"
    lr: float = 0.01
    weight_decay: float = 0.005
    momentum: float = 0.9

@dataclass
class SchedulerConfig:
    step_size : int = 10
    gamma : float = 0.1

@dataclass
class TrainConfig:
    batch_size: int = 128
    num_epochs: int = 20
    num_workers: int = 6

@dataclass
class Config:
    project_name: str = "vgg16"
    author: str = "wanghongyu"
    version: str = "2.0"
    
    data: DataConfig = field(default_factory=DataConfig)
    log_path: LogConfig = field(default_factory=LogConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler : SchedulerConfig = field(default_factory = SchedulerConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    
    @classmethod
    def from_toml(cls,path:str) -> "Config":
        with open(path,'rb') as f:
            file = toml.load(f)
        config = cls()
        
        for key in ["project_name", "author", "version"]:
            if key in file:
                setattr(config,key,file[key])
        
        if "data" in file:
            config.data = DataConfig(**file["data"])
        if "log_path" in file:
            config.log_path = LogConfig(**file["log_path"])
        if "model" in file:
            config.model = ModelConfig(**file["model"])
        if "optimizer" in file:
            config.optimizer = OptimizerConfig(**file["optimizer"])
        if "train" in file:
            config.train = TrainConfig(**file["train"])
        if "scheduler" in file:
            config.scheduler = SchedulerConfig(**file["scheduler"])
        
        return config
    
def build_features(config: Config) -> nn.Module:
    features = nn.Sequential()
    conv = config.model.conv_blocks
    last_output = config.data.input_channels
    conv_num = 1
    pool_num = 1
    for layer in conv:
        for item in layer:
            if isinstance(item,int):
                features.add_module(
                            f"conv{conv_num}",
                           nn.Conv2d(last_output,item,
                                     kernel_size = 3,stride = 1,padding = 1)
                )
                last_output = item
                features.add_module(
                        f"bn{conv_num}",
                        nn.BatchNorm2d(last_output)
                )
                features.add_module(
                        f"relu{conv_num}",
                        nn.ReLU()
                )
                conv_num += 1
            elif item =='m':
                features.add_module(
                        f"maxpool{pool_num}",
                        nn.MaxPool2d(kernel_size = 2,stride = 2)
                )
                pool_num += 1
    return features

def build_fc(config: Config) -> nn.Module:
    full_connect = nn.Sequential()
    
    full_connect.add_module(
            "adaptive_avg_pool",
            nn.AdaptiveAvgPool2d((1, 1))
    )
    
    full_connect.add_module(
            "flatten",
            nn.Flatten()
    )
    
    in_features = 512 * 1 * 1
    layer_count = 1
    
    for layer in config.model.fc_layers:
        for item in layer:
            if isinstance(item, int):
                full_connect.add_module(
                        f"fc{layer_count}",
                        nn.Linear(in_features, item)
                )
                in_features = item
                break
        
        for item in layer:
            if item == "r":
                full_connect.add_module(
                        f"relu{layer_count}",
                        nn.ReLU(inplace=True)
                )
            elif item == "d":
                dropout_rate = 0.5
                for next_item in layer:
                    if isinstance(next_item, float):
                        dropout_rate = next_item
                        break
                
                full_connect.add_module(
                        f"dropout{layer_count}",
                        nn.Dropout(dropout_rate)
                )
        
        layer_count += 1
    
    return full_connect

class vgg16(nn.Module):
    def __init__(self,config : Config):
        super().__init__()
        self.feature = build_features(config)
        self.fc = build_fc(config)

    def forward(self,x):
        x = self.feature(x)
        x = self.fc(x)
        
        return x
    
    
def init_(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


def load_cifar100(config : Config):
    data_root = config.data.data_path
    batch_size = config.train.batch_size
    num_workers = config.train.num_workers

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    
    train_set = torchvision.datasets.CIFAR100(
            root = data_root,
            train = True,
            download = True,
            transform = transform_train
    )
    
    test_set = torchvision.datasets.CIFAR100(
            root = data_root,
            train = False,
            download = True,
            transform = transform_test
    )
    
    train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size = batch_size,
            shuffle = True,
            pin_memory = True,
            num_workers = num_workers
    )
    
    test_loader = torch.utils.data.DataLoader(
            test_set,
            batch_size = batch_size,
            shuffle = False,
            pin_memory = True,
            num_workers = num_workers
    )
    
    return train_loader, test_loader


def main() ->None:
    config = Config().from_toml('../Config/vgg_config.toml')
    
    train_loader,test_loader = load_cifar100(config)
    
    model = vgg16(config)
    model.apply(init_)
    
    loss_fn = nn.CrossEntropyLoss()
    
    lr = config.optimizer.lr
    weight_decay = config.optimizer.weight_decay
    mt = config.optimizer.momentum
    
    optimizer = torch.optim.Adam(
            model.parameters(),
            lr = lr,
            weight_decay = weight_decay
    )
    
    step_size = config.scheduler.step_size
    ggamma = config.scheduler.gamma
    
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
            step_size = step_size,
            gamma = ggamma
    )
    
    num_epochs = config.train.num_epochs
    train_log_path = config.log_path.train_log_save_path
    test_log_path = config.log_path.test_log_save_path
    model_save = config.data.model_save
    
    print(f"\n开始训练，共{num_epochs}个epoch...")
    print(f"训练日志文件: {train_log_path}")
    print(f"测试日志文件: {test_log_path}")
    print(f"模型将保存到: {model_save}")

    start_time = time.time()
    
    train_losses, test_accuracies, best_accuracy = tp.train_model(
            model = model,
            train_loader = train_loader,
            test_loader = test_loader,
            loss_fn = loss_fn,
            optimizer = optimizer,
            scheduler = scheduler,
            epochs = num_epochs,
            save_path = model_save,
            log_file = train_log_path
    )
    
    end_time = time.time()
    
    print("\n使用最佳模型进行测试...")
    predictions, accuracy, total, correct, labels = tp.predict_model(
            model = model,
            data_loader = test_loader,
            model_path = model_save,
            log_file = test_log_path
    )
    
    training_time = end_time - start_time
    
    print("\n" + "=" * 60)
    print("训练完成！")
    print("=" * 60)
    print(f"训练用时: {training_time:.2f}秒 ({training_time / 60:.2f}分钟)")
    print(f"训练过程中最佳准确率: {best_accuracy:.2f}%")
    print(f"最终测试准确率: {accuracy:.2f}%")
    print(f"正确数/总数: {correct}/{total}")

if __name__ == '__main__':
    main()
