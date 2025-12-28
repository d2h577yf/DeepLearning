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
        loss_fn: Optional[nn.Module] = None,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        lr: float = 0.001,
        weight_decay: float = 0.0,
        epochs: int = 10,
        device: str = 'cuda',
        save_path: str = 'best_model.pth',
        log_file: Optional[str] = None,
        use_amp: bool = True,  # 新增：是否使用混合精度
        log_interval: int = 50  # 新增：日志间隔
) -> Tuple[List[float], List[float], float]:
    
    logger = Logger(log_file)
    
    # 修正设备选择逻辑
    device = torch.device('cuda' if torch.cuda.is_available() and device == 'cuda' else 'cpu')
    logger.log(f"使用设备: {device}")
    
    # 启用cudNN优化
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        logger.log("已启用cudNN自动优化")
    
    model = model.to(device)
    
    loss_fn = loss_fn or nn.CrossEntropyLoss()
    optimizer = optimizer or optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = scheduler or optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    # 混合精度训练
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and device.type == 'cuda')
    
    # 计算模型参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.log(f"模型参数: 总计 {total_params:,}, 可训练 {trainable_params:,}")
    
    logger.log("\n训练配置:")
    logger.log(f"  学习率: {lr}")
    logger.log(f"  权重衰减: {weight_decay}")
    logger.log(f"  训练轮数: {epochs}")
    logger.log(f"  批大小: {train_loader.batch_size}")
    logger.log(f"  训练集大小: {len(train_loader.dataset)}")
    logger.log(f"  测试集大小: {len(test_loader.dataset)}")
    logger.log(f"  混合精度训练: {use_amp and device.type == 'cuda'}")
    logger.log(f"  日志间隔: 每{log_interval}个batch")
    
    train_losses, test_accuracies = [], []
    best_accuracy = 0.0
    
    for epoch in range(epochs):
        logger.log(f"\nEpoch {epoch+1}/{epochs}")
        logger.log("-" * 30)
        
        model.train()
        running_loss = correct = total = 0
        
        # 使用tqdm但设置更小的更新间隔
        with tqdm(train_loader, desc="训练", mininterval=1.0) as pbar:
            for batch_idx, (images, labels) in enumerate(pbar):
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                
                # 混合精度训练
                with torch.cuda.amp.autocast(enabled=use_amp and device.type == 'cuda'):
                    outputs = model(images)
                    loss = loss_fn(outputs, labels)
                
                optimizer.zero_grad(set_to_none=True)  # 更快的梯度清零
                
                if use_amp and device.type == 'cuda':
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                
                running_loss += loss.item()
                
                # 每log_interval个batch才计算准确率，减少开销
                if batch_idx % log_interval == 0 or batch_idx == len(train_loader) - 1:
                    with torch.no_grad():
                        _, predicted = torch.max(outputs.data, 1)
                        batch_total = labels.size(0)
                        batch_correct = (predicted == labels).sum().item()
                        total += batch_total
                        correct += batch_correct
                        
                        current_acc = 100 * batch_correct / batch_total
                        avg_acc = 100 * correct / total if total > 0 else 0
                        
                        pbar.set_postfix({
                            'loss': f'{loss.item():.4f}',
                            'batch_acc': f'{current_acc:.1f}%',
                            'avg_acc': f'{avg_acc:.1f}%'
                        })
        
        # 计算平均训练损失和准确率
        avg_train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct / total if total > 0 else 0
        train_losses.append(avg_train_loss)
        
        # 学习率调整
        if scheduler is not None:
            scheduler.step()
        
        # 测试
        model.eval()
        test_correct = test_total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc="测试", leave=False):
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                
                # 测试时也使用混合精度
                with torch.cuda.amp.autocast(enabled=use_amp and device.type == 'cuda'):
                    outputs = model(images)
                
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
        
        test_accuracy = 100 * test_correct / test_total
        test_accuracies.append(test_accuracy)
        
        # 保存最佳模型
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'loss': avg_train_loss,
                'accuracy': best_accuracy,
            }, save_path)
            logger.log(f"保存最佳模型: {best_accuracy:.2f}%", to_console=False)
        
        logger.log(f"训练损失: {avg_train_loss:.4f}, 训练准确率: {train_accuracy:.2f}%")
        logger.log(f"测试准确率: {test_accuracy:.2f}%")
        if scheduler:
            current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else optimizer.param_groups[0]['lr']
            logger.log(f"当前学习率: {current_lr:.6f}")
    
    logger.log(f"\n训练完成!")
    logger.log(f"最佳测试准确率: {best_accuracy:.2f}%")
    
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
                
                # 如果有标签，计算准确率
                if has_labels:
                    total += labels.size(0)
                    correct += (predictions == labels).sum().item()
                    pbar.set_postfix({'acc': f'{100*correct/total:.1f}%'})
    
    predictions = torch.cat(all_predictions)
    
    if has_labels:
        labels = torch.cat(all_labels)
        # 确保 accuracy 是 Python float 而不是 Tensor
        accuracy = 100.0 * correct / total if total > 0 else 0.0
        accuracy_float = float(accuracy)  # 转换为 Python float
        logger.log(f"测试结果: {accuracy_float:.2f}% ({correct}/{total})")
        model.cpu()
        return predictions, accuracy_float, total, correct, labels
    else:
        logger.log(f"预测完成，总样本数: {len(predictions)}")
        model.cpu()
        return predictions,
