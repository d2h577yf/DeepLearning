import torch
import matplotlib.pyplot as plt
import numpy as np

# 修复d2l显示问题
import matplotlib_inline
import IPython.display as display

def use_svg_display():
    """使用svg格式显示绘图"""
    matplotlib_inline.backend_inline.set_matplotlib_formats('svg')

from d2l import torch as d2l
d2l.use_svg_display = use_svg_display

# 模型定义
def softmax(X):
    """softmax函数"""
    X_exp = torch.exp(X)
    partition = X_exp.sum(dim=1, keepdim=True)
    return X_exp / partition

def net(X):
    """网络前向传播"""
    return softmax(torch.matmul(X.reshape(-1, W.shape[0]), W) + b)

def cross_entropy(y_hat, y):
    """交叉熵损失函数"""
    return - torch.log(y_hat[range(len(y_hat)), y])

def accuracy(y_hat, y):
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

class Accumulator:
    """在n个变量上累加"""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def evaluate_accuracy(net, data_iter):
    """计算在指定数据集上模型的精度"""
    if isinstance(net, torch.nn.Module):
        net.eval()
    metric = Accumulator(2)  # 正确预测数、预测总数
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

class Animator:
    """在动画中绘制数据（命令行适配版）"""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        # 增量地绘制多条线
        if legend is None:
            legend = []

        # 使用d2l的设置
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]

        # 使用lambda函数捕获参数
        self.config_axes = lambda: d2l.set_axes(
                self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # 向图表中添加多个数据点
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)

        # 清空并重新绘制
        self.axes[0].cla()
        for x_data, y_data, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x_data, y_data, fmt)
        self.config_axes()

        # 重绘图形
        self.fig.canvas.draw()
        plt.pause(0.01)  # 短暂暂停以允许更新

    def show(self):
        """显示最终图表"""
        plt.show()

def train_epoch_ch3(net, train_iter, loss, updater):
    """训练模型一个迭代周期"""
    # 将模型设置为训练模式
    if isinstance(net, torch.nn.Module):
        net.train()

    # 训练损失总和、训练准确度总和、样本数
    metric = Accumulator(3)

    for X, y in train_iter:
        # 计算梯度并更新参数
        y_hat = net(X)
        l = loss(y_hat, y)

        if isinstance(updater, torch.optim.Optimizer):
            # 使用PyTorch内置的优化器和损失函数
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            # 使用定制的优化器和损失函数
            l.sum().backward()
            updater(X.shape[0])

        # 修复警告：使用.detach()分离梯度
        metric.add(l.sum().detach().item(), accuracy(y_hat, y), y.numel())

    # 返回训练损失和训练精度
    return metric[0] / metric[2], metric[1] / metric[2]

def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
    """训练模型"""
    print(f"开始训练，共 {num_epochs} 个epoch")

    # 创建动画器
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])

    train_losses = []
    train_accs = []
    test_accs = []

    for epoch in range(num_epochs):
        # 训练一个epoch
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)

        # 评估测试集
        test_acc = evaluate_accuracy(net, test_iter)

        # 记录数据
        train_loss, train_acc = train_metrics
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)

        # 更新图表
        animator.add(epoch + 1, train_metrics + (test_acc,))

        # 打印进度
        print(f"Epoch {epoch+1}/{num_epochs}: "
              f"loss={train_loss:.4f}, train_acc={train_acc:.3f}, test_acc={test_acc:.3f}")

    # 训练结束后显示图表
    animator.show()

    # 验证结果
    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, train_loss
    assert 1 >= train_acc > 0.7, train_acc
    assert 1 >= test_acc > 0.7, test_acc

    return train_losses, train_accs, test_accs

def updater(batch_size):
    """优化器更新函数"""
    return d2l.sgd([W, b], lr, batch_size)

def predict_ch3(net, test_iter, n=6):
    """预测标签"""
    # 获取一批测试数据
    for X, y in test_iter:
        break

    # 获取真实标签和预测标签
    trues = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))

    # 创建子图显示图像
    fig, axes = plt.subplots(1, n, figsize=(n*3, 3))
    if n == 1:
        axes = [axes]

    for i in range(n):
        ax = axes[i]
        # 显示图像
        img = X[i].reshape(28, 28).numpy()
        ax.imshow(img, cmap='gray')

        # 设置标题（绿色表示正确，红色表示错误）
        true_label = trues[i]
        pred_label = preds[i]
        color = 'green' if true_label == pred_label else 'red'
        ax.set_title(f"True: {true_label}\nPred: {pred_label}", color=color)
        ax.axis('off')

    plt.tight_layout()
    plt.show()

    # 打印准确率统计
    correct = sum(1 for true, pred in zip(trues[:n], preds[:n]) if true == pred)
    print(f"\n前{n}个样本的预测准确率: {correct}/{n} ({correct/n:.1%})")

def print_model_summary():
    """打印模型参数摘要"""
    print("=" * 60)
    print("模型参数摘要:")
    print(f"  输入维度: {num_inputs}")
    print(f"  输出维度: {num_outputs}")
    print(f"  权重W形状: {W.shape}")
    print(f"  偏置b形状: {b.shape}")
    print(f"  总参数量: {W.numel() + b.numel():,}")
    print(f"  学习率: {lr}")
    print("=" * 60)

if __name__ == '__main__':
    # 设置随机种子以确保可重复性
    torch.manual_seed(42)

    # 加载数据
    batch_size = 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    print(f"数据加载完成!")
    print(f"  训练集批次大小: {batch_size}")
    print(f"  测试集批次大小: {batch_size}")

    # 模型参数初始化
    num_inputs = 784  # 28x28
    num_outputs = 10  # 10个类别

    # 初始化权重和偏置
    W = torch.normal(0, 0.01, (num_inputs, num_outputs), requires_grad=True)
    b = torch.zeros(num_outputs, requires_grad=True)
    lr = 0.1

    # 打印模型信息
    print_model_summary()

    # 训练模型
    num_epochs = 9
    print(f"\n开始训练，共 {num_epochs} 个epoch...")
    train_losses, train_accs, test_accs = train_ch3(
            net, train_iter, test_iter, cross_entropy, num_epochs, updater
    )

    # 打印最终结果
    print("\n" + "=" * 60)
    print("训练完成!")
    print(f"最终训练损失: {train_losses[-1]:.4f}")
    print(f"最终训练准确率: {train_accs[-1]:.3f}")
    print(f"最终测试准确率: {test_accs[-1]:.3f}")
    print("=" * 60)

    # 可视化一些预测结果
    print("\n可视化预测结果...")
    predict_ch3(net, test_iter, n=10)

    # 最终评估
    final_accuracy = evaluate_accuracy(net, test_iter)
    print(f"\n模型在完整测试集上的准确率: {final_accuracy:.2%}")