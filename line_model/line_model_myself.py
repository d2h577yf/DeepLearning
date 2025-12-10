import numpy as np
import matplotlib.pyplot as plt
import random
import torch
from d2l import torch as d2l
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Songti SC', 'STHeiti']
plt.rcParams['axes.unicode_minus'] = False

def data_iter(batch_size, features, labels):
    n_samples = len(features)
    indices = list(range(n_samples))
    random.shuffle(indices)
    for i in range(0,n_samples,batch_size):
        batch_indices = torch.tensor(indices[i:min(i + batch_size,n_samples)])
        yield features[batch_indices], labels[batch_indices]

def linreg(X, w, b):  #@save
    """线性回归模型"""
    return torch.matmul(X, w) + b

def squared_loss(y_hat, y):  #@save
    """均方损失"""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

def sgd(params, lr, batch_size):  #@save
    """小批量随机梯度下降"""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)
# print(len(true_w))
# plt.figure(figsize=(8, 6))
# plt.scatter(features[:, 1].detach().numpy(),
#             labels.detach().numpy(),
#             1)
#
# plt.title('Feature 2 vs Labels')
# plt.xlabel('Feature 2')
# plt.ylabel('Labels')
# plt.grid(True)
# plt.show()
#
# batch_size = 10
#
# for X, y in data_iter(batch_size, features, labels):
#     print(X, '\n', y)
#     break

# 初始化参数
# w = torch.normal(0, 0.01, size=(2,1), requires_grad=True)
w = torch.zeros(true_w.shape,requires_grad=True)
b = torch.zeros(1, requires_grad=True)
features, labels = d2l.synthetic_data(true_w,true_b,1000)

batch_size = 10
lr = 0.03
num_epochs = 20
net = linreg
loss = squared_loss

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y)
        l.sum().backward()
        sgd([w, b], lr, batch_size)
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')


# def visualize_training():
#     # 真实参数和目标
#     true_w = np.array([2.0, -3.4])
#     true_b = 4.2
#
#     # 初始参数（猜测的）
#     w = np.array([0.5, -1.0])
#     b = 1.0
#
#     # 模拟几个训练步骤
#     steps = 10
#     loss_history = []
#     w_history = []
#     b_history = []
#
#     for step in range(steps):
#         # 模拟计算损失（真实值与预测值的差异）
#         current_loss = np.random.rand() * (1 - step / steps)  # 损失逐渐减小
#         loss_history.append(current_loss)
#
#         # 模拟参数更新（向真实值靠近）
#         w += (true_w - w) * 0.3
#         b += (true_b - b) * 0.3
#
#         w_history.append(w.copy())
#         b_history.append(b)
#
#     # 可视化
#     fig, axes = plt.subplots(2, 2, figsize=(12, 8))
#
#     # 损失下降曲线
#     axes[0, 0].plot(loss_history, 'b-o')
#     axes[0, 0].set_title('Loss下降曲线')
#     axes[0, 0].set_xlabel('训练步数')
#     axes[0, 0].set_ylabel('损失值')
#     axes[0, 0].grid(True)
#
#     # 权重w1的收敛过程
#     axes[0, 1].plot([w[0] for w in w_history], 'r-o', label='w[0]')
#     axes[0, 1].axhline(y=true_w[0], color='r', linestyle='--', label='真实w[0]')
#     axes[0, 1].set_title('权重w[0]收敛过程')
#     axes[0, 1].set_xlabel('训练步数')
#     axes[0, 1].set_ylabel('w[0]值')
#     axes[0, 1].legend()
#     axes[0, 1].grid(True)
#
#     # 权重w2的收敛过程
#     axes[1, 0].plot([w[1] for w in w_history], 'g-o', label='w[1]')
#     axes[1, 0].axhline(y=true_w[1], color='g', linestyle='--', label='真实w[1]')
#     axes[1, 0].set_title('权重w[1]收敛过程')
#     axes[1, 0].set_xlabel('训练步数')
#     axes[1, 0].set_ylabel('w[1]值')
#     axes[1, 0].legend()
#     axes[1, 0].grid(True)
#
#     # 偏置b的收敛过程
#     axes[1, 1].plot(b_history, 'b-o', label='b')
#     axes[1, 1].axhline(y=true_b, color='b', linestyle='--', label='真实b')
#     axes[1, 1].set_title('偏置b收敛过程')
#     axes[1, 1].set_xlabel('训练步数')
#     axes[1, 1].set_ylabel('b值')
#     axes[1, 1].legend()
#     axes[1, 1].grid(True)
#
#     plt.tight_layout()
#     plt.show()
#
#
# visualize_training()

learning_rates = [0.001, 0.01, 0.03, 0.1, 0.3]
for lr in learning_rates:
    # 重新初始化参数
    w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)

    losses = []
    for epoch in range(10):
        for X, y in data_iter(batch_size, features, labels):
            l = loss(net(X, w, b), y)
            l.sum().backward()
            sgd([w, b], lr, batch_size)

        with torch.no_grad():
            epoch_loss = loss(net(features, w, b), labels).mean().item()
            losses.append(epoch_loss)

    print(f"学习率 {lr}: 最终损失={losses[-1]:.6f}, 损失曲线={losses}")