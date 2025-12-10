import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional
import numpy as np


# ==================== 工具类 ====================

class Accumulator:
    """用于累积多个变量的和"""

    def __init__(self, n: int):
        self.data = [0.0] * n

    def add(self, *args):
        """添加值到累积器"""
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        """重置累积器"""
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class TrainingMonitor:
    """训练过程监控器，用于可视化训练指标"""

    def __init__(self, figsize = (10, 4)):
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize = figsize)
        self.train_losses = []
        self.train_accs = []
        self.test_accs = []
        self.epochs = []

        # 设置子图
        self.ax1.set_xlabel('Epoch')
        self.ax1.set_ylabel('Loss')
        self.ax1.set_title('Training Loss')

        self.ax2.set_xlabel('Epoch')
        self.ax2.set_ylabel('Accuracy')
        self.ax2.set_title('Training & Test Accuracy')
        self.ax2.set_ylim(0, 1)

        plt.ion()  # 开启交互模式

    def update(self, epoch: int, train_loss: float, train_acc: float, test_acc: Optional[float] = None):
        """更新监控器"""
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.train_accs.append(train_acc)

        if test_acc is not None:
            self.test_accs.append(test_acc)

        # 清除并重绘
        self.ax1.clear()
        self.ax2.clear()

        # 绘制损失
        self.ax1.plot(self.epochs, self.train_losses, 'b-', label = 'Train Loss')
        self.ax1.set_xlabel('Epoch')
        self.ax1.set_ylabel('Loss')
        self.ax1.set_title('Training Loss')
        self.ax1.legend()
        self.ax1.grid(True, alpha = 0.3)

        # 绘制准确率
        self.ax2.plot(self.epochs, self.train_accs, 'g-', label = 'Train Acc')
        if self.test_accs:
            self.ax2.plot(self.epochs, self.test_accs, 'r-', label = 'Test Acc')
        self.ax2.set_xlabel('Epoch')
        self.ax2.set_ylabel('Accuracy')
        self.ax2.set_title('Accuracy Curves')
        self.ax2.set_ylim(0, 1)
        self.ax2.legend()
        self.ax2.grid(True, alpha = 0.3)

        self.fig.tight_layout()
        plt.draw()
        plt.pause(0.1)

    def save(self, filename: str = 'training_history.png'):
        """保存训练历史图像"""
        self.fig.savefig(filename, dpi = 300, bbox_inches = 'tight')
        plt.ioff()
        plt.show()

    def close(self):
        """关闭图像"""
        plt.ioff()
        plt.close()


# ==================== 核心函数 ====================

def accuracy(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """计算预测准确率"""
    if y_hat.dim() > 1 and y_hat.shape[1] > 1:
        # 多分类：取最大概率的类别
        y_hat = y_hat.argmax(dim = 1)
    cmp = y_hat.type(y.dtype) == y
    return cmp.type(y.dtype).sum()


def evaluate_accuracy(net: nn.Module, data_iter: DataLoader, device: torch.device = None) -> float:
    """评估模型在数据集上的准确率"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net.eval()
    metric = Accumulator(2)  # [正确数, 总数]

    with torch.no_grad():
        for X, y in data_iter:
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            metric.add(accuracy(y_hat, y), y.numel())

    return metric[0] / metric[1]


def train_epoch(net: nn.Module, train_iter: DataLoader, loss_fn: nn.Module,
                optimizer: torch.optim.Optimizer, device: torch.device = None) -> Tuple[float, float]:
    """训练一个epoch"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net.train()
    metric = Accumulator(3)  # [总损失, 正确数, 样本数]

    for X, y in train_iter:
        X, y = X.to(device), y.to(device)

        # 前向传播
        y_hat = net(X)
        loss = loss_fn(y_hat, y)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 累积指标
        metric.add(loss.item() * y.size(0), accuracy(y_hat, y), y.size(0))

    # 返回平均损失和准确率
    return metric[0] / metric[2], metric[1] / metric[2]


def train_model(net: nn.Module, train_iter: DataLoader, test_iter: DataLoader,
                loss_fn: nn.Module, optimizer: torch.optim.Optimizer,
                num_epochs: int = 10, device: torch.device = None,
                save_path: str = None, show_plot: bool = True) -> dict:
    """
    训练模型主函数

    Args:
        net: 神经网络模型
        train_iter: 训练数据迭代器
        test_iter: 测试数据迭代器
        loss_fn: 损失函数
        optimizer: 优化器
        num_epochs: 训练轮数
        device: 训练设备
        save_path: 模型保存路径
        show_plot: 是否显示训练曲线

    Returns:
        dict: 包含训练历史的字典
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = net.to(device)
    history = {'train_loss': [], 'train_acc': [], 'test_acc': []}

    if show_plot:
        monitor = TrainingMonitor()

    print(f"开始训练，使用设备: {device}")
    print(f"{'Epoch':<10} {'Train Loss':<15} {'Train Acc':<15} {'Test Acc':<15}")
    print("-" * 60)

    for epoch in range(1, num_epochs + 1):
        # 训练一个epoch
        train_loss, train_acc = train_epoch(net, train_iter, loss_fn, optimizer, device)

        # 评估测试集
        test_acc = evaluate_accuracy(net, test_iter, device)

        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)

        # 打印进度
        print(f"{epoch:<10} {train_loss:<15.4f} {train_acc:<15.4f} {test_acc:<15.4f}")

        # 更新监控器
        if show_plot:
            monitor.update(epoch, train_loss, train_acc, test_acc)

    # 保存模型
    if save_path:
        torch.save({
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'history': history
        }, save_path)
        print(f"\n模型已保存到: {save_path}")

    # 保存训练图像
    if show_plot:
        monitor.save('training_history.png')
        monitor.close()

    print("\n训练完成!")
    return history


def predict(net: nn.Module, data_iter: DataLoader, num_samples: int = 6,
            class_names: List[str] = None, device: torch.device = None):
    """
    对数据集进行预测并可视化结果

    Args:
        net: 训练好的模型
        data_iter: 数据迭代器
        num_samples: 显示的样本数量
        class_names: 类别名称列表
        device: 推理设备
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net.eval()

    # 获取一个batch的数据
    for X, y in data_iter:
        break

    X, y = X.to(device), y.to(device)

    # 预测
    with torch.no_grad():
        y_hat = net(X)
        preds = y_hat.argmax(dim = 1)

    # 限制显示数量
    num_samples = min(num_samples, X.size(0))

    # 如果没有提供类别名称，使用数字标签
    if class_names is None:
        class_names = [str(i) for i in range(10)]

    # 可视化预测结果
    fig, axes = plt.subplots(1, num_samples, figsize = (3 * num_samples, 3))
    if num_samples == 1:
        axes = [axes]

    for i in range(num_samples):
        img = X[i].cpu().squeeze().numpy()

        # 如果是单通道图像，确保正确的维度
        if len(img.shape) == 3:
            img = img[0]  # 取第一个通道

        axes[i].imshow(img, cmap = 'gray')
        axes[i].set_title(f"True: {class_names[y[i].item()]}\nPred: {class_names[preds[i].item()]}")

        # 标记预测错误为红色
        if y[i].item() != preds[i].item():
            axes[i].title.set_color('red')

        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

    # 计算并显示batch准确率
    correct = (preds == y).sum().item()
    total = y.size(0)
    print(f"当前batch准确率: {correct}/{total} ({correct / total:.2%})")

    return preds, y


def load_and_predict(net: nn.Module, checkpoint_path: str, data_iter: DataLoader,
                     class_names: List[str] = None, device: torch.device = None):
    """
    加载训练好的模型并进行预测
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location = device)
    net.load_state_dict(checkpoint['model_state_dict'])
    net.to(device)

    print(f"模型从 {checkpoint_path} 加载成功")

    # 进行预测
    return predict(net, data_iter, class_names = class_names, device = device)


# ==================== 使用示例 ====================

def example_usage():
    """使用示例"""
    import torchvision
    import torchvision.transforms as transforms

    # 1. 准备数据
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # 下载FashionMNIST数据集
    train_dataset = torchvision.datasets.FashionMNIST(
            root = '../data', train = True, download = True, transform = transform)
    test_dataset = torchvision.datasets.FashionMNIST(
            root = '../data', train = False, download = True, transform = transform)

    train_loader = DataLoader(train_dataset, batch_size = 64, shuffle = True)
    test_loader = DataLoader(test_dataset, batch_size = 64, shuffle = False)

    # 2. 定义模型
    class SimpleCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(28 * 28, 256),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(256, 10)
            )

        def forward(self, x):
            return self.net(x)

    # 3. 初始化模型、损失函数、优化器
    model = SimpleCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

    # 4. 训练模型
    history = train_model(
            net = model,
            train_iter = train_loader,
            test_iter = test_loader,
            loss_fn = criterion,
            optimizer = optimizer,
            num_epochs = 10,
            save_path = 'model.pth',
            show_plot = True
    )

    # 5. 进行预测
    # FashionMNIST类别名称
    fashion_classes = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
                       'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    print("\n进行预测...")
    predict(model, test_loader, num_samples = 8, class_names = fashion_classes)

    return history
