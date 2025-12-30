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
    
from pprint import pprint


def build_features(config: Config) -> nn.Module:
    conv = config.model.conv_blocks
    last_output = config.data.input_channels
    for layer in conv:
        for output_channel in layer:
            pprint(output_channel)
        
config = Config.from_toml('Config/vgg_config.toml')

build_features(config)
