import torch
from torch import nn
from torch import optim
from torch.utils import data
from tqdm import tqdm
from typing import Optional, Tuple, List, Dict, Any
import time
import os

class Logger:
    """简单的日志记录器"""
    def __init__(self, log_file: str = None):
        self.log_file = log_file
        if log_file:
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir)
            with open(log_file, 'w', encoding='utf-8') as f:
                f.write(f"训练日志 - {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 50 + "\n\n")
    
    def log(self, message: str, to_console: bool = True):
        """记录日志"""
        if to_console:
            print(message)
        if self.log_file:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(f"[{time.strftime('%H:%M:%S')}] {message}\n")


def train_model(
        model: nn.Module,
        train_loader: data.DataLoader,
        test_loader: data.DataLoader,
        loss_fn: Optional[nn.Module],
        optimizer: Optional[optim.Optimizer],
        scheduler: Optional[Any] = None,
        epochs: int = 10,
        device: str = 'cuda',
        save_path: str = 'best_model.pth',
        log_file: Optional[str] = None
) -> Tuple[List[float], List[float], float]:
    """
    训练模型
    
    参数:
        model: 要训练的模型
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
        loss_fn: 损失函数，默认为CrossEntropyLoss
        optimizer: 优化器，默认为Adam
        scheduler: 学习率调度器，默认为StepLR
        lr: 学习率
        weight_decay: 权重衰减
        epochs: 训练轮数
        device: 训练设备
        save_path: 模型保存路径
        log_file: 日志文件路径，如果为None则不记录到文件
        
    返回:
        train_losses: 训练损失列表
        test_accuracies: 测试准确率列表
        best_accuracy: 最佳测试准确率
    """
    
    logger = Logger(log_file)
    
    device = torch.device(device if device == 'cuda' and torch.cuda.is_available() else 'cpu')
    logger.log(f"使用设备: {device}")
    
    model = model.to(device)
    
    loss_fn = loss_fn or nn.CrossEntropyLoss()
    optimizer = optimizer
    scheduler = scheduler
    
    logger.log("\n训练配置:")
    logger.log(f"  学习率: {optimizer.param_groups[0]['lr']}")
    logger.log(f"  权重衰减: {optimizer.param_groups[0]['weight_decay']}")
    logger.log(f"  训练轮数: {epochs}")
    logger.log(f"  批大小: {train_loader.batch_size}")
    logger.log(f"  训练集大小: {len(train_loader.dataset)}")
    logger.log(f"  测试集大小: {len(test_loader.dataset)}")
    
    train_losses, test_accuracies = [], []
    best_accuracy = 0.0
    
    for epoch in range(epochs):
        logger.log(f"\nEpoch {epoch+1}/{epochs}")
        logger.log("-" * 30)
        
        model.train()
        running_loss = correct = total = 0
        
        with tqdm(train_loader, desc="训练") as pbar:
            for images, labels in pbar:
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                loss = loss_fn(outputs, labels)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100*correct/total:.1f}%'
                })
                
        if scheduler:
            scheduler.step()
        
        avg_train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct / total
        train_losses.append(avg_train_loss)
        
        model.eval()
        test_correct = test_total = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
        
        test_accuracy = 100 * test_correct / test_total
        test_accuracies.append(test_accuracy)
        
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_train_loss,
                'accuracy': best_accuracy,
            }, save_path)
            logger.log(f"保存最佳模型: {best_accuracy:.2f}%", to_console=False)
        
        logger.log(f"训练损失: {avg_train_loss:.4f}, 训练准确率: {train_accuracy:.2f}%")
        logger.log(f"测试准确率: {test_accuracy:.2f}%")
    
    logger.log(f"\n训练完成!")
    logger.log(f"最佳测试准确率: {best_accuracy:.2f}%")
    
    model.cpu()
    return train_losses, test_accuracies, best_accuracy


def predict_model(
        model: nn.Module,
        data_loader: data.DataLoader,
        device: str = 'cuda',
        model_path: Optional[str] = None,
        log_file: Optional[str] = None
) -> Tuple:
    """
    使用模型进行预测
    
    参数:
        model: 用于预测的模型
        data_loader: 数据加载器（如果有标签会自动计算准确率）
        device: 预测设备
        model_path: 模型权重路径
        log_file: 日志文件路径
        
    返回:
        如果数据加载器有标签，返回:
            predictions: 所有预测结果
            accuracy: 准确率（Python float）
            total: 总样本数（Python int）
            correct: 正确预测数（Python int）
            labels: 所有真实标签
        如果无标签，返回:
            predictions: 所有预测结果
    """
    
    logger = Logger(log_file)
    
    device = torch.device(device if device == 'cuda' and torch.cuda.is_available() else 'cpu')
    logger.log(f"使用设备: {device}")
    
    if model_path:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.log(f"加载模型权重: {model_path}")
    
    model.to(device).eval()
    all_predictions = []
    all_labels = []
    correct = total = 0
    
    has_labels = True
    try:
        sample_batch = next(iter(data_loader))
        has_labels = len(sample_batch) == 2
    except:
        has_labels = False
    
    with torch.no_grad():
        desc = "测试" if has_labels else "预测"
        with tqdm(data_loader, desc=desc) as pbar:
            for batch in pbar:
                if has_labels:
                    images, labels = batch
                    images, labels = images.to(device), labels.to(device)
                    all_labels.append(labels.cpu())
                else:
                    images = batch[0] if isinstance(batch, tuple) else batch
                    images = images.to(device)
                
                outputs = model(images)
                _, predictions = torch.max(outputs, 1)
                all_predictions.append(predictions.cpu())
                
                if has_labels:
                    total += labels.size(0)
                    correct += (predictions == labels).sum().item()
                    pbar.set_postfix({'acc': f'{100*correct/total:.1f}%'})
    
    predictions = torch.cat(all_predictions)
    
    if has_labels:
        labels = torch.cat(all_labels)
        accuracy = 100.0 * correct / total if total > 0 else 0.0
        accuracy_float = float(accuracy)  # 转换为 Python float
        logger.log(f"测试结果: {accuracy_float:.2f}% ({correct}/{total})")
        model.cpu()
        return predictions, accuracy_float, total, correct, labels
    else:
        logger.log(f"预测完成，总样本数: {len(predictions)}")
        model.cpu()
        return predictions,
