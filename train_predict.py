import torch
from torch import nn
from torch import optim
from torch.utils import data
from tqdm import tqdm

def train_model(model: nn.Module,
                train_loader: data.DataLoader,
                test_loader: data.DataLoader,
                lr: float,
                epochs: int = 10,
                device: str = 'cuda'):  # 新增device参数
    
    if device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    print(f"使用设备: {device}")
    
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    train_losses = []
    test_accuracies = []
    best_accuracy = 0.0
    
    for epoch in range(epochs):
        print(f"\n{'=' * 50}")
        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"{'=' * 50}")
        
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        train_bar = tqdm(train_loader, desc=f"训练", leave=False)
        for images, labels in train_bar:
            images, labels = images.to(device), labels.to(device)
            
            labels_hat = model(images)
            loss = criterion(labels_hat, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(labels_hat.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            train_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * correct / total:.2f}%'
            })
        
        scheduler.step()
        
        avg_train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct / total
        
        model.eval()
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                
                y_hat = model(images)
                _, predicted = torch.max(y_hat.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
        
        test_accuracy = 100 * test_correct / test_total
        
        train_losses.append(avg_train_loss)
        test_accuracies.append(test_accuracy)
        
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_model_state = model.state_dict().copy()
            torch.save({
                'epoch': epoch,
                'model_state_dict': best_model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_train_loss,
                'accuracy': best_accuracy,
            }, 'best_lenet_model.pth')
            print(f"✅ 保存最佳模型，准确率: {best_accuracy:.2f}%")
        
        print(f"训练损失: {avg_train_loss:.4f}, 训练准确率: {train_accuracy:.2f}%")
        print(f"测试准确率: {test_accuracy:.2f}%")
    
    print(f"\n训练完成！最佳测试准确率: {best_accuracy:.2f}%")
    
    model = model.to('cpu')
    return train_losses, test_accuracies, best_accuracy


def test_model(model: nn.Module,
               test_loader: data.DataLoader,
               device: str = 'cuda'):
    
    if device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    model = model.to(device)
    
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for img, label in test_loader:
            img, label = img.to(device), label.to(device)
            
            outputs = model(img)
            _, predicted = torch.max(outputs.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
    
    accuracy = 100 * correct / total
    
    model = model.to('cpu')
    return accuracy, total, correct

