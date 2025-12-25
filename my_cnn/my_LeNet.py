# Library import
import time
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn
from torch.utils import data
import train_predict as tp

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Songti SC', 'STHeiti']
plt.rcParams['axes.unicode_minus'] = False

def load_mnist(batch_size : int , show_some_information : bool = False) -> tuple[data.DataLoader, data.DataLoader]:
    """
    加载MNIST数据集
    
    参数:
        batch_size: 每个批次的样本数量
        
    返回:
        train_loader: 训练集数据加载器
        test_loader: 测试集数据加载器
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Pad(2),
        transforms.Normalize((0.1307,), (0.3081,)) # 计算整个MNIST数据集的统计得到的均值 ≈ 0.1307，标准差 ≈ 0.3081
    ])
    
    train_dataset = torchvision.datasets.MNIST(
            root = '../data',
            train = True,
            download = True,
            transform = transform
    )
    
    test_dataset = torchvision.datasets.MNIST(
            root = '../data',
            train = False,
            download = True,
            transform = transform
    )
    
    train_loader = data.DataLoader(  # 使用data.DataLoader
            train_dataset,
            batch_size = batch_size,
            shuffle = True,
            num_workers = 4
    )
    
    test_loader = data.DataLoader(  # 使用data.DataLoader
            test_dataset,
            batch_size = batch_size,
            shuffle = False,
            num_workers = 4
    )
    
    if show_some_information :
        small_loader = data.DataLoader(
                train_dataset,
                batch_size = 5,
                shuffle = True
        )
        
        images,labels = next(iter(small_loader))
    
        print(f"图像形状: {images.shape}")  # [5, 1, 28, 28]
        print(f"标签形状: {labels.shape}")
        
        fig,axes = plt.subplots(1,5,figsize=(12,3))
        
        for i in range(5):
            img = images[i].squeeze().numpy()
            axes[i].imshow(img, cmap = 'gray')
            axes[i].set_title(f'Label: {labels[i].item()}')
            axes[i].axis('off')
            
        plt.suptitle('MNIST样本展示')
        plt.tight_layout()
        plt.show()
            
    return train_loader, test_loader

class MyLeNetOriginal(nn.Module):
    def __init__(self):
        super().__init__()
        self.C1 = nn.Conv2d(1, 6, kernel_size=5)
        self.S2 = TrainableAvgPool2d(6, kernel_size=2)
        self.C3 = nn.Conv2d(6, 16, kernel_size=5)
        self.S4 = TrainableAvgPool2d(16, kernel_size=2)
        self.C5 = nn.Conv2d(16, 120, kernel_size=5)
        self.F6 = nn.Linear(120, 84)
        self.output = nn.Linear(84, 10)
        
        self.A = 1.7159
        self.S = 2 / 3
    
    def scaled_tanh(self, x):
        return self.A * torch.tanh(self.S * x)
    
    def forward(self, x):
        x = self.scaled_tanh(self.C1(x))
        x = self.scaled_tanh(self.S2(x))
        x = self.scaled_tanh(self.C3(x))
        x = self.scaled_tanh(self.S4(x))
        x = self.scaled_tanh(self.C5(x))  #
        x = x.view(x.size(0), -1)  # [batch, 120]
        x = self.scaled_tanh(self.F6(x))  # [batch, 84]
        x = self.output(x)  # [batch, 10]
        
        return x

class TrainableAvgPool2d(nn.Module):
    def __init__(self, num_features, kernel_size):
        super().__init__()
        self.num_features = num_features
        self.pool_size = kernel_size

        self.weight = nn.Parameter(torch.normal(mean=0.0, std=0.01, size=(num_features,)))
        self.bias = nn.Parameter(torch.zeros(num_features))
        
        self.pool = nn.AvgPool2d(kernel_size=kernel_size, stride=kernel_size)
    
    def forward(self, x):
        x = self.pool(x)
        weight = self.weight.view(1, -1, 1, 1)
        bias = self.bias.view(1, -1, 1, 1)
        x = weight * x + bias
        return x

class MyLeNetModern(nn.Module):
    def __init__(self):
        super().__init__()
        self.C1 = nn.Conv2d(1, 6, kernel_size=5)
        self.S2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.C3 = nn.Conv2d(6, 16, kernel_size=5)
        self.S4 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.C5 = nn.Conv2d(16, 120, kernel_size=5)
        self.F6 = nn.Linear(120, 84)
        self.output = nn.Linear(84, 10)
    
        self.relu = nn.PReLU()
        
        self.dropout = nn.Dropout(0.25)
    
    def forward(self, x):
        x = self.C1(x)
        x = self.relu(x)
        x = self.S2(x)
        
        x = self.C3(x)
        x = self.relu(x)
        x = self.S4(x)
        
        x = self.C5(x)
        x = self.relu(x)
        
        x = x.view(x.size(0), -1)
        
        x = self.F6(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.output(x)
        
        return x


# 现已放到train_predict文件中
# def train_model(model: nn.Module,
#                 train_loader: data.DataLoader,
#                 test_loader: data.DataLoader,
#                 lr: float,
#                 epochs: int = 10):
#
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr)
#     scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
#
#     train_losses = []
#     test_accuracies = []
#     best_accuracy = 0.0
#
#     for epoch in range(epochs):
#         print(f"\n{'=' * 50}")
#         print(f"Epoch {epoch + 1}/{epochs}")
#         print(f"{'=' * 50}")
#
#         model.train()
#         running_loss = 0.0
#         correct = 0
#         total = 0
#
#         train_bar = tqdm(train_loader, desc=f"训练", leave=False)
#         for images, labels in train_bar:
#             labels_hat = model(images)
#             loss = criterion(labels_hat, labels)
#
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#
#             running_loss += loss.item()
#             _, predicted = torch.max(labels_hat.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#
#             train_bar.set_postfix({
#                 'loss': f'{loss.item():.4f}',
#                 'acc': f'{100 * correct / total:.2f}%'
#             })
#
#         scheduler.step()
#
#         avg_train_loss = running_loss / len(train_loader)
#         train_accuracy = 100 * correct / total
#
#         model.eval()
#         test_correct = 0
#         test_total = 0
#
#         with torch.no_grad():
#             for images, labels in test_loader:
#                 y_hat = model(images)
#                 _, predicted = torch.max(y_hat.data, 1)
#                 test_total += labels.size(0)
#                 test_correct += (predicted == labels).sum().item()
#
#         test_accuracy = 100 * test_correct / test_total
#
#         train_losses.append(avg_train_loss)
#         test_accuracies.append(test_accuracy)
#
#         if test_accuracy > best_accuracy:
#             best_accuracy = test_accuracy
#             best_model_state = model.state_dict().copy()
#             torch.save({
#                 'epoch': epoch,
#                 'model_state_dict': best_model_state,
#                 'optimizer_state_dict': optimizer.state_dict(),
#                 'loss': avg_train_loss,
#                 'accuracy': best_accuracy,
#             }, 'best_lenet_model.pth')
#             print(f"✅ 保存最佳模型，准确率: {best_accuracy:.2f}%")
#
#         print(f"训练损失: {avg_train_loss:.4f}, 训练准确率: {train_accuracy:.2f}%")
#         print(f"测试准确率: {test_accuracy:.2f}%")
#
#     print(f"\n训练完成！最佳测试准确率: {best_accuracy:.2f}%")
#     return train_losses, test_accuracies, best_accuracy
#
# def test_model(model : nn.Module,
#                test_loader : data.DataLoader):
#     model.eval()
#     correct = 0
#     total = 0
#
#     with torch.no_grad():
#         for img,label in test_loader:
#             outputs = model(img)
#             _, predicted = torch.max(outputs.data, 1)
#             total += label.size(0)
#             correct += (predicted == label).sum().item()
#
#     accuracy = 100 * correct / total
#     return accuracy, total, correct

def main(model_choice : str = 'm'):
    batch_size : int = 64
    train_loader,test_loader = load_mnist(batch_size)
    
    
    if model_choice == 'o':
        print("原始模型:")
        model = MyLeNetOriginal()
    elif model_choice == 'm':
        print("现代模型:")
        model = MyLeNetModern()
        def init_weights(m):
            if isinstance(m,nn.Conv2d) or isinstance(m,nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        model.apply(init_weights)
    elif model_choice == 'd':
        print("D2L中的模型:")
        model = nn.Sequential(
                nn.Conv2d(1,6,kernel_size = 5),nn.Sigmoid(),
                nn.AvgPool2d(kernel_size = 2,stride = 2),
                nn.Conv2d(6,16,kernel_size = 5),nn.Sigmoid(),
                nn.AvgPool2d(kernel_size = 2, stride = 2),
                nn.Flatten(),
                nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
                nn.Linear(120, 84), nn.Sigmoid(),
                nn.Linear(84, 10)
        )
    
    start_time = time.time()
    train_losses, test_accuracies, best_accuracy = tp.train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=20,
        lr=0.001
    )
    
    end_time = time.time()
    
    final_accuracy, total, correct = tp.test_model(model, test_loader)
    print(f"\n训练用时: {end_time - start_time:.2f}秒")
    print(f"最终测试准确率: {final_accuracy:.2f}%")
    print(f"正确数/总数: {correct}/{total}")
    
if __name__ == '__main__':
    main('m')