import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Tuple, List, Optional, Dict, Union, Callable
import time
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


class ConsoleMonitor:
    """终端训练监控器，用于在终端显示训练进度和指标"""

    def __init__(self, num_epochs: int, show_progress_bar: bool = True):
        self.num_epochs = num_epochs
        self.show_progress_bar = show_progress_bar
        self.start_time = time.time()
        self.best_test_acc = 0.0
        self.best_epoch = 0

    def print_header(self):
        """打印表头"""
        print(f"{'Epoch':<10} {'Train Loss':<15} {'Train Acc':<15} {'Test Acc':<15} {'Time':<15}")
        print("-" * 70)

    def print_epoch(self, epoch: int, train_loss: float, train_acc: float, test_acc: float):
        """打印每个epoch的结果"""
        epoch_time = time.time() - self.start_time

        # 更新最佳准确率
        if test_acc > self.best_test_acc:
            self.best_test_acc = test_acc
            self.best_epoch = epoch

        print(f"{epoch:<10} {train_loss:<15.4f} {train_acc:<15.4f} {test_acc:<15.4f} {epoch_time:<15.2f}s")

        # 显示进度条（可选）
        if self.show_progress_bar and epoch > 0:
            self._print_progress_bar(epoch)

    def _print_progress_bar(self, epoch: int):
        """显示训练进度条"""
        progress = int((epoch / self.num_epochs) * 40)
        bar = "[" + "=" * progress + ">" + " " * (40 - progress) + "]"
        percent = (epoch / self.num_epochs) * 100
        print(f"进度: {bar} {percent:.1f}% ({epoch}/{self.num_epochs})")

    def print_summary(self, history: Dict):
        """打印训练总结"""
        print("\n" + "=" * 70)
        print("训练完成!")
        print(f"总训练时间: {time.time() - self.start_time:.2f}秒")
        print(f"最佳测试准确率: {self.best_test_acc:.4f} (第{self.best_epoch}个epoch)")

        # 打印最终结果
        print(f"\n最终结果:")
        print(f"训练损失: {history['train_loss'][-1]:.4f}")
        print(f"训练准确率: {history['train_acc'][-1]:.4f}")
        print(f"测试准确率: {history['test_acc'][-1]:.4f}")

        # 打印改进情况
        if len(history['train_acc']) > 1:
            train_improvement = history['train_acc'][-1] - history['train_acc'][0]
            test_improvement = history['test_acc'][-1] - history['test_acc'][0]
            print(f"\n改进情况:")
            print(f"训练准确率提升: {train_improvement:+.4f}")
            print(f"测试准确率提升: {test_improvement:+.4f}")

            # 检查过拟合
            train_test_gap = history['train_acc'][-1] - history['test_acc'][-1]
            if train_test_gap > 0.15:  # 如果训练准确率比测试准确率高15%以上
                print(f"⚠️  注意: 可能存在过拟合 (训练-测试差距: {train_test_gap:.4f})")
            elif train_test_gap < -0.05:  # 如果测试准确率比训练准确率高5%以上
                print(f"⚠️  注意: 可能存在欠拟合 (训练-测试差距: {train_test_gap:.4f})")

    def print_checkpoint_saved(self, save_path: str):
        """打印模型保存信息"""
        print(f"💾 模型已保存到: {save_path}")


# ==================== 模型类型检测 ====================

def is_functional_model(model) -> bool:
    """
    判断是否为函数式模型（函数 + 参数列表）

    Args:
        model: 模型对象

    Returns:
        bool: 是否为函数式模型
    """
    if isinstance(model, tuple) and len(model) == 2:
        # 如果是(函数, 参数列表)的元组
        return callable(model[0]) and isinstance(model[1], list)
    elif callable(model):
        # 如果是函数，检查是否有相关的全局参数
        return True
    else:
        return False


def prepare_functional_model(model, device: torch.device):
    """
    准备函数式模型进行训练

    Args:
        model: 函数式模型
        device: 训练设备

    Returns:
        准备好训练的函数式模型
    """
    if isinstance(model, tuple):
        # 模型是(函数, 参数列表)形式
        forward_fn, params = model
        # 将参数移动到设备
        params = [param.to(device) for param in params]
        return forward_fn, params
    else:
        # 模型是函数，需要在外部定义参数
        return model


# ==================== 核心函数 ====================

def accuracy(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """计算预测准确率"""
    if y_hat.dim() > 1 and y_hat.shape[1] > 1:
        # 多分类：取最大概率的类别
        y_hat = y_hat.argmax(dim=1)
    cmp = y_hat.type(y.dtype) == y
    return cmp.type(y.dtype).sum()


def evaluate_accuracy(model, data_iter: DataLoader,
                      device: torch.device = None) -> float:
    """评估模型在数据集上的准确率"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 判断模型类型
    is_functional = is_functional_model(model)

    metric = Accumulator(2)  # [正确数, 总数]

    with torch.no_grad():
        for X, y in data_iter:
            X, y = X.to(device), y.to(device)

            # 根据模型类型进行前向传播
            if is_functional:
                if isinstance(model, tuple):
                    forward_fn, params = model
                    y_hat = forward_fn(X, *params)
                else:
                    # 函数式模型，直接调用
                    y_hat = model(X)
            elif isinstance(model, nn.Module):
                model.eval()
                y_hat = model(X)
            else:
                raise ValueError(f"不支持的模型类型: {type(model)}")

            metric.add(accuracy(y_hat, y), y.numel())

    return metric[0] / metric[1]


def train_epoch(model, train_iter: DataLoader, loss_fn: nn.Module,
                optimizer: torch.optim.Optimizer, device: torch.device = None) -> Tuple[float, float]:
    """训练一个epoch"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 判断模型类型
    is_functional = is_functional_model(model)

    if isinstance(model, nn.Module):
        model.train()

    metric = Accumulator(3)  # [总损失, 正确数, 样本数]

    for batch_idx, (X, y) in enumerate(train_iter):
        X, y = X.to(device), y.to(device)

        # 前向传播
        if is_functional:
            if isinstance(model, tuple):
                forward_fn, params = model
                y_hat = forward_fn(X, *params)
            else:
                y_hat = model(X)
        elif isinstance(model, nn.Module):
            y_hat = model(X)
        else:
            raise ValueError(f"不支持的模型类型: {type(model)}")

        loss = loss_fn(y_hat, y)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 累积指标
        metric.add(loss.item() * y.size(0), accuracy(y_hat, y), y.size(0))

        # 每10个batch打印一次进度
        if batch_idx % 10 == 0 and batch_idx > 0:
            avg_loss = metric[0] / metric[2]
            avg_acc = metric[1] / metric[2]
            print(f"  Batch {batch_idx}/{len(train_iter)} - Loss: {avg_loss:.4f}, Acc: {avg_acc:.4f}")

    # 返回平均损失和准确率
    return metric[0] / metric[2], metric[1] / metric[2]


def train_model(model, train_iter: DataLoader, test_iter: DataLoader,
                loss_fn: nn.Module, optimizer: torch.optim.Optimizer,
                num_epochs: int = 10, device: torch.device = None,
                save_path: Optional[str] = None, show_progress_bar: bool = True,
                validate_every: int = 1) -> Dict:
    """
    训练模型主函数

    Args:
        model: 神经网络模型，可以是nn.Module、函数或(函数, 参数)元组
        train_iter: 训练数据迭代器
        test_iter: 测试数据迭代器
        loss_fn: 损失函数
        optimizer: 优化器
        num_epochs: 训练轮数
        device: 训练设备
        save_path: 模型保存路径
        show_progress_bar: 是否显示进度条
        validate_every: 每隔多少个epoch验证一次

    Returns:
        dict: 包含训练历史的字典
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 判断模型类型
    is_functional = is_functional_model(model)
    is_module = isinstance(model, nn.Module)

    if is_functional:
        print("🎯 检测到函数式模型")
        if isinstance(model, tuple):
            forward_fn, params = model
            params = [param.to(device) for param in params]
            model = (forward_fn, params)
    elif is_module:
        print("🧠 检测到标准nn.Module模型")
        model = model.to(device)
    else:
        raise ValueError(f"不支持的模型类型: {type(model)}")

    history = {'train_loss': [], 'train_acc': [], 'test_acc': [], 'epoch_times': []}

    # 初始化控制台监控器
    monitor = ConsoleMonitor(num_epochs, show_progress_bar)

    print(f"🚀 开始训练，使用设备: {device}")
    print(f"📊 总epoch数: {num_epochs}, 批量大小: {train_iter.batch_size}")
    monitor.print_header()

    for epoch in range(1, num_epochs + 1):
        epoch_start_time = time.time()

        # 训练一个epoch
        train_loss, train_acc = train_epoch(model, train_iter, loss_fn, optimizer, device)

        # 记录epoch时间
        epoch_time = time.time() - epoch_start_time
        history['epoch_times'].append(epoch_time)

        # 每隔validate_every个epoch评估一次测试集
        if epoch % validate_every == 0 or epoch == num_epochs:
            test_acc = evaluate_accuracy(model, test_iter, device)
        else:
            test_acc = history['test_acc'][-1] if history['test_acc'] else 0.0

        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)

        # 打印epoch结果
        monitor.print_epoch(epoch, train_loss, train_acc, test_acc)

        # 保存模型检查点
        if save_path and epoch % 5 == 0:
            checkpoint_path = f"{save_path}_epoch_{epoch}.pth"
            # 根据模型类型保存
            if is_module:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'test_acc': test_acc
                }, checkpoint_path)
            else:
                print(f"⚠️  函数式模型无法保存为标准的PyTorch模型格式，跳过保存")

            monitor.print_checkpoint_saved(checkpoint_path)

    # 训练完成，打印总结
    monitor.print_summary(history)

    # 最终保存模型
    if save_path:
        if is_module:
            final_save_path = f"{save_path}_final.pth" if not save_path.endswith('.pth') else save_path
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history,
                'num_epochs': num_epochs,
                'final_train_acc': history['train_acc'][-1],
                'final_test_acc': history['test_acc'][-1]
            }, final_save_path)
            print(f"💾 最终模型已保存到: {final_save_path}")
        else:
            print("⚠️  函数式模型无法保存为标准的PyTorch模型格式")
            print("   请手动保存模型参数")

    print("\n✅ 训练完成!")
    return history


def predict(model, data_iter: DataLoader,
            num_samples: int = 10, class_names: List[str] = None,
            device: torch.device = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    对数据集进行预测

    Args:
        model: 训练好的模型
        data_iter: 数据迭代器
        num_samples: 显示的样本数量
        class_names: 类别名称列表
        device: 推理设备

    Returns:
        Tuple: (预测结果, 真实标签)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 判断模型类型
    is_functional = is_functional_model(model)

    if isinstance(model, nn.Module):
        model.eval()

    # 获取一个batch的数据
    for X, y in data_iter:
        break

    X, y = X.to(device), y.to(device)

    # 预测
    with torch.no_grad():
        if is_functional:
            if isinstance(model, tuple):
                forward_fn, params = model
                y_hat = forward_fn(X, *params)
            else:
                y_hat = model(X)
        elif isinstance(model, nn.Module):
            y_hat = model(X)
        else:
            raise ValueError(f"不支持的模型类型: {type(model)}")

        preds = y_hat.argmax(dim=1) if y_hat.dim() > 1 else y_hat

    # 限制显示数量
    num_samples = min(num_samples, X.size(0))

    # 如果没有提供类别名称，使用数字标签
    if class_names is None:
        class_names = [str(i) for i in range(10)]

    # 打印预测结果
    print(f"\n🔍 预测结果 (显示前{num_samples}个样本):")
    print(f"{'样本':<10} {'预测':<15} {'真实':<15} {'状态':<15}")
    print("-" * 55)

    correct_count = 0
    for i in range(num_samples):
        pred_label = preds[i].item()
        true_label = y[i].item()

        # 获取类别名称
        pred_name = class_names[pred_label] if pred_label < len(class_names) else str(pred_label)
        true_name = class_names[true_label] if true_label < len(class_names) else str(true_label)

        is_correct = pred_label == true_label
        status = "✅ 正确" if is_correct else "❌ 错误"

        if is_correct:
            correct_count += 1

        print(f"{i+1:<10} {pred_name:<15} {true_name:<15} {status:<15}")

    # 计算并显示batch准确率
    total_correct = (preds == y).sum().item()
    total = y.size(0)
    accuracy = total_correct / total

    print(f"\n📊 当前batch统计:")
    print(f"  样本总数: {total}")
    print(f"  正确预测: {total_correct}")
    print(f"  准确率: {accuracy:.2%}")
    print(f"  显示样本正确率: {correct_count}/{num_samples} ({correct_count/num_samples:.2%})")

    return preds, y


def print_model_info(model):
    """打印模型信息"""
    print("\n📋 模型信息:")

    is_functional = is_functional_model(model)

    if is_functional:
        print("  类型: 函数式模型")
        if isinstance(model, tuple):
            forward_fn, params = model
            print(f"  参数数量: {len(params)}")
            for i, param in enumerate(params):
                print(f"    参数{i}: shape={param.shape}, dtype={param.dtype}, requires_grad={param.requires_grad}")
    elif isinstance(model, nn.Module):
        print("  类型: 标准nn.Module")
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  总参数数量: {total_params:,}")
        print(f"  可训练参数: {trainable_params:,}")
        print(f"  不可训练参数: {total_params - trainable_params:,}")

        # 打印层信息
        print("\n  层信息:")
        for name, module in model.named_children():
            num_params = sum(p.numel() for p in module.parameters())
            print(f"    {name}: {module.__class__.__name__}, 参数: {num_params:,}")
    else:
        print(f"  类型: {type(model)}")
