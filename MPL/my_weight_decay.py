import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Songti SC', 'STHeiti']
plt.rcParams['axes.unicode_minus'] = False

# 1. 生成线性回归数据
def generate_linear_data(num_samples = 20, num_inputs = 200, noise = 0.01):
    """生成高维线性数据（故意过拟合设置）"""
    # 真实权重：只有前几个维度有效，后面都是噪声
    true_w = torch.cat([torch.tensor([2.0, -3.4]), torch.zeros(num_inputs - 2)])
    true_b = 4.2

    # 生成数据
    X = torch.randn(num_samples, num_inputs)
    y = X @ true_w + true_b + torch.randn(num_samples) * noise

    # 创建数据加载器（简化的迭代器）
    train_iter = [(X[:15], y[:15])]  # 15个训练样本
    test_iter = [(X[15:], y[15:])]  # 5个测试样本

    return train_iter, test_iter, true_w


# 2. 训练函数（简洁版）
def train_simple(wd = 0.0, lr = 0.003, epochs = 100, show_plot = True):
    """训练线性回归模型，带权重衰减"""
    num_inputs = 200
    train_iter, test_iter, true_w = generate_linear_data(num_inputs = num_inputs)

    # 创建模型（一层线性层）
    net = nn.Linear(num_inputs, 1)

    # 初始化权重（正态分布）
    nn.init.normal_(net.weight, mean = 0, std = 0.01)
    nn.init.zeros_(net.bias)

    # 损失函数
    loss_fn = nn.MSELoss()

    # 优化器：只对权重使用权重衰减，偏置不使用
    trainer = optim.SGD([
        {"params": net.weight, 'weight_decay': wd},  # 权重有衰减
        {"params": net.bias}  # 偏置无衰减
    ], lr = lr)

    # 记录损失
    train_losses = []
    test_losses = []

    print(f"训练开始: wd={wd}, lr={lr}")
    print("-" * 50)

    for epoch in range(epochs):
        net.train()

        # 训练步骤
        for X, y in train_iter:
            trainer.zero_grad()
            y_pred = net(X)
            loss = loss_fn(y_pred, y.unsqueeze(1))
            loss.backward()
            trainer.step()

        # 每5轮记录一次损失
        if (epoch + 1) % 5 == 0:
            # 训练损失
            with torch.no_grad():
                train_pred = net(train_iter[0][0])
                train_loss = loss_fn(train_pred, train_iter[0][1].unsqueeze(1))

                # 测试损失
                test_pred = net(test_iter[0][0])
                test_loss = loss_fn(test_pred, test_iter[0][1].unsqueeze(1))

            train_losses.append((epoch + 1, train_loss.item()))
            test_losses.append((epoch + 1, test_loss.item()))

            print(f"Epoch {epoch + 1:3d}: 训练损失={train_loss:.4f}, 测试损失={test_loss:.4f}")

    # 计算最终权重范数
    w_norm = net.weight.norm().item()
    print(f"\n最终权重L2范数: {w_norm:.4f}")
    print(f"权重衰减效果: {w_norm < 5.0}")

    # 绘制损失曲线
    if show_plot:
        plot_losses(train_losses, test_losses, wd)

    return net, w_norm


# 3. 绘图函数
def plot_losses(train_losses, test_losses, wd):
    """绘制训练和测试损失"""
    epochs = [x[0] for x in train_losses]
    train_vals = [x[1] for x in train_losses]
    test_vals = [x[1] for x in test_losses]

    plt.figure(figsize = (10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_vals, 'b-', label = '训练损失')
    plt.plot(epochs, test_vals, 'r-', label = '测试损失')
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.title(f'权重衰减 wd={wd}')
    plt.legend()
    plt.grid(True, alpha = 0.3)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_vals, 'b-', label = '训练损失')
    plt.plot(epochs, test_vals, 'r-', label = '测试损失')
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.yscale('log')  # 对数坐标
    plt.title(f'损失（对数坐标）')
    plt.legend()
    plt.grid(True, alpha = 0.3)

    plt.tight_layout()
    plt.show()


# 4. 主函数：对比不同权重衰减值
def compare_weight_decay():
    """对比不同权重衰减值的效果"""
    wd_values = [0, 0.001, 0.01, 0.1, 1.0]
    results = {}

    print("=== 权重衰减对比实验 ===")
    print(f"模型：单层线性回归，输入维度=200，训练样本=15，测试样本=5")
    print("=" * 60)

    for wd in wd_values:
        print(f"\n>>> 测试 wd = {wd}")
        model, w_norm = train_simple(wd = wd, lr = 0.003, epochs = 100, show_plot = False)
        results[wd] = w_norm

        # 显示权重统计
        with torch.no_grad():
            weight = model.weight.squeeze()
            print(f"  权重范围: [{weight.min():.4f}, {weight.max():.4f}]")
            print(f"  绝对值>0.1的权重比例: {(weight.abs() > 0.1).sum().item()}/{len(weight)}")

    # 绘制权重范数对比
    print("\n" + "=" * 60)
    print("权重衰减效果总结:")

    wds = list(results.keys())
    norms = list(results.values())

    plt.figure(figsize = (8, 4))
    plt.plot(wds, norms, 'bo-', linewidth = 2, markersize = 8)
    plt.xlabel('权重衰减 (wd)')
    plt.ylabel('权重L2范数')
    plt.title('权重衰减 vs 权重大小')
    plt.grid(True, alpha = 0.3)

    # 添加标注
    for i, (wd, norm) in enumerate(zip(wds, norms)):
        plt.annotate(f'{norm:.2f}', xy = (wd, norm), xytext = (5, 5),
                     textcoords = 'offset points', fontsize = 9)

    plt.show()

    return results


# 5. 单次运行示例
if __name__ == "__main__":
    # 选项1：单个实验（带可视化）
    print("=== 单个权重衰减实验 ===")
    model, w_norm = train_simple(wd = 0.01, lr = 0.003, epochs = 100, show_plot = True)

    # 选项2：对比实验（取消注释运行）
    print("\n=== 权重衰减对比实验 ===")
    results = compare_weight_decay()

    # 选项3：观察过拟合现象（无权重衰减）
    print("\n=== 过拟合现象演示（无权重衰减） ===")
    model_no_wd, _ = train_simple(wd = 0.0, lr = 0.003, epochs = 100, show_plot = True)