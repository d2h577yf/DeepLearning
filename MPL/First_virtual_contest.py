"""
房价预测项目 - 深度学习实践
基于Kaggle房价预测比赛的完整实现
包含数据下载、预处理、模型训练和评估
"""

# ============================================
# 第一部分：导入必要的库
# ============================================

# 数据处理和下载相关库
import hashlib
import os
import tarfile
import zipfile
from typing import Dict, Tuple, Optional

# 可视化和工具库
import matplotlib.pyplot as plt
import pandas as pd
# 数据科学和机器学习库
import requests
import torch
from torch import nn
from torch.utils import data

# ============================================
# 第二部分：数据下载工具
# ============================================

# 数据仓库字典
DATA_HUB: Dict[str, Tuple[str, str]] = {}
DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'


def download(name: str, cache_dir: str = os.path.join('..', 'data')) -> str:
    """
    从DATA_HUB下载文件，返回本地文件名

    Args:
        name: 数据集名称
        cache_dir: 缓存目录路径

    Returns:
        str: 本地文件路径

    Raises:
        AssertionError: 如果数据集名称不在DATA_HUB中
    """
    assert name in DATA_HUB, f"{name} 不存在于 {DATA_HUB}"

    url, sha1_hash = DATA_HUB[name]
    os.makedirs(cache_dir, exist_ok=True)
    fname = os.path.join(cache_dir, url.split('/')[-1])

    # 检查缓存
    if os.path.exists(fname):
        sha1 = hashlib.sha1()
        with open(fname, 'rb') as f:
            while True:
                data = f.read(1048576)  # 读取1MB
                if not data:
                    break
                sha1.update(data)

        if sha1.hexdigest() == sha1_hash:
            return fname  # 命中缓存

    # 下载文件
    print(f'正在从 {url} 下载 {fname}...')
    r = requests.get(url, stream=True, verify=True)
    with open(fname, 'wb') as f:
        f.write(r.content)

    return fname


def download_extract(name: str, folder: Optional[str] = None) -> str:
    """
    下载并解压zip/tar文件

    Args:
        name: 数据集名称
        folder: 解压目标文件夹

    Returns:
        str: 解压后的目录路径
    """
    fname = download(name)
    base_dir = os.path.dirname(fname)
    data_dir, ext = os.path.splitext(fname)

    # 根据扩展名选择解压方式
    if ext == '.zip':
        with zipfile.ZipFile(fname, 'r') as fp:
            fp.extractall(base_dir)
    elif ext in ('.tar', '.gz'):
        with tarfile.open(fname, 'r') as fp:
            fp.extractall(base_dir)
    else:
        raise ValueError('只有 zip/tar 文件可以被解压缩')

    return os.path.join(base_dir, folder) if folder else data_dir


def download_all() -> None:
    """下载DATA_HUB中的所有文件"""
    for name in DATA_HUB:
        download(name)


# ============================================
# 第三部分：数据预处理函数
# ============================================

def preprocess_data(train_data: pd.DataFrame, test_data: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    预处理房价预测数据

    Args:
        train_data: 训练数据
        test_data: 测试数据

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 训练特征、测试特征、训练标签
    """
    print("\n开始数据预处理...")

    # 合并特征
    all_features = pd.concat([
        train_data.iloc[:, 1:-1],  # 排除 Id 和 SalePrice
        test_data.iloc[:, 1:]      # 排除 Id
    ])

    print(f'合并特征形状: {all_features.shape}')

    # 标准化数值型特征
    numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
    all_features[numeric_features] = all_features[numeric_features].apply(
            lambda x: (x - x.mean()) / (x.std())
    )

    # 填充数值型特征的缺失值
    all_features[numeric_features] = all_features[numeric_features].fillna(0)

    # 独热编码分类型特征
    all_features = pd.get_dummies(all_features, dummy_na=True)

    # 转换bool类型为float
    bool_features = all_features.dtypes[all_features.dtypes == 'bool'].index
    all_features[bool_features] = all_features[bool_features].astype(float)

    # 转换为PyTorch张量
    n_train = train_data.shape[0]

    train_features = torch.tensor(
            all_features[:n_train].values,
            dtype=torch.float32
    )

    test_features = torch.tensor(
            all_features[n_train:].values,
            dtype=torch.float32
    )

    train_labels = torch.tensor(
            train_data['SalePrice'].values.reshape(-1, 1),
            dtype=torch.float32
    )

    print(f'处理后的训练特征形状: {train_features.shape}')
    print(f'处理后的测试特征形状: {test_features.shape}')
    print(f'训练标签形状: {train_labels.shape}')

    return train_features, test_features, train_labels


# ============================================
# 第四部分：模型定义函数
# ============================================

def get_net(features_num: int) -> nn.Module:
    """
    创建简单的线性回归模型

    Args:
        features_num: 输入特征数量

    Returns:
        nn.Module: 神经网络模型
    """
    net = nn.Sequential(
            nn.Linear(features_num, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
    )
    return net


# ============================================
# 第五部分：训练和评估函数
# ============================================

def log_rmse(net: nn.Module, features: torch.Tensor, labels: torch.Tensor, loss_fn: nn.Module) -> float:
    """
    计算对数均方根误差

    Args:
        net: 神经网络模型
        features: 输入特征
        labels: 真实标签
        loss_fn: 损失函数

    Returns:
        float: 对数均方根误差
    """
    clipped_preds = torch.clamp(net(features), 1, float('inf'))
    rmse = torch.sqrt(
            loss_fn(
                    torch.log(clipped_preds),
                    torch.log(labels)
            )
    )
    return rmse.item()


def train(
        net: nn.Module,
        train_f: torch.Tensor,
        train_l: torch.Tensor,
        test_f: torch.Tensor,
        test_l: torch.Tensor,
        num_epochs: int,
        learning_rate: float,
        weight_decay: float,
        batch_size: int,
        loss_fn: nn.Module
) -> Tuple[list[float], list[float]]:
    """
    训练神经网络模型

    Args:
        net: 神经网络模型
        train_f: 训练特征
        train_l: 训练标签
        test_f: 测试特征
        test_l: 测试标签
        num_epochs: 训练轮数
        learning_rate: 学习率
        weight_decay: 权重衰减系数
        batch_size: 批次大小
        loss_fn: 损失函数

    Returns:
        Tuple[list[float], list[float]]:
            train_ls: 训练集log_RMSE历史
            test_ls: 测试集log_RMSE历史
    """
    train_ls, test_ls = [], []

    dataset = data.TensorDataset(train_f, train_l)
    train_iter = data.DataLoader(dataset, batch_size, shuffle=True)

    optimizer = torch.optim.Adam(
            net.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
    )

    for epoch in range(num_epochs):
        for X, y in train_iter:
            l = loss_fn(net(X), y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()

        train_ls.append(log_rmse(net, train_f, train_l, loss_fn))

        if test_l is not None:
            test_ls.append(log_rmse(net, test_f, test_l, loss_fn))

        print(f"Epoch {epoch+1}/{num_epochs}, "
              f"Train logRMSE: {train_ls[-1]:.4f}, "
              f"Test logRMSE: {test_ls[-1] if test_ls else 'N/A'}")

    return train_ls, test_ls


# ============================================
# 第六部分：K折交叉验证函数
# ============================================

def get_k_fold_data(k: int, i: int, X: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    获取K折交叉验证的数据划分

    Args:
        k: 折数
        i: 当前折索引
        X: 特征张量
        y: 标签张量

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: 训练特征、训练标签、验证特征、验证标签
    """
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat([X_train, X_part], 0)
            y_train = torch.cat([y_train, y_part], 0)
    return X_train, y_train, X_valid, y_valid


def k_fold(
        k: int,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        num_epochs: int,
        learning_rate: float,
        weight_decay: float,
        batch_size: int,
        features_num: int
) -> Tuple[float, float]:
    """
    执行K折交叉验证

    Args:
        k: 折数
        X_train: 训练特征
        y_train: 训练标签
        num_epochs: 训练轮数
        learning_rate: 学习率
        weight_decay: 权重衰减系数
        batch_size: 批次大小
        features_num: 特征数量

    Returns:
        Tuple[float, float]: 平均训练log_RMSE, 平均验证log_RMSE
    """
    train_l_sum, valid_l_sum = 0, 0

    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        net = get_net(features_num)
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate,
                                   weight_decay, batch_size, nn.MSELoss())

        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]

        # 只绘制第一折的曲线
        if i == 0:
            epochs = list(range(1, num_epochs + 1))
            plt.figure(figsize=(10, 6))
            plt.plot(epochs, train_ls, label='train', marker='o')
            plt.plot(epochs, valid_ls, label='valid', marker='s')
            plt.xlabel('epoch')
            plt.ylabel('log rmse')
            plt.xlim(1, num_epochs)
            plt.legend()
            plt.yscale('log')  # 设置y轴为对数刻度
            plt.grid(True)
            plt.title('Training and Validation Log RMSE (Fold 1)')
            plt.show()

        print(f'折{i+1}，训练log rmse: {train_ls[-1]:.4f}, 验证log rmse: {valid_ls[-1]:.4f}')

    return train_l_sum / k, valid_l_sum / k


# ============================================
# 第七部分：训练和预测函数
# ============================================

def train_and_pred(
        train_features: torch.Tensor,
        test_features: torch.Tensor,
        train_labels: torch.Tensor,
        test_data: pd.DataFrame,
        num_epochs: int,
        learning_rate: float,
        weight_decay: float,
        batch_size: int,
        features_num: int
) -> pd.DataFrame:
    """
    训练模型并生成预测

    Args:
        train_features: 训练特征
        test_features: 测试特征
        train_labels: 训练标签
        test_data: 原始测试数据
        num_epochs: 训练轮数
        learning_rate: 学习率
        weight_decay: 权重衰减系数
        batch_size: 批次大小
        features_num: 特征数量

    Returns:
        pd.DataFrame: 提交结果DataFrame
    """
    net = get_net(features_num)
    train_ls, _ = train(net, train_features, train_labels, None, None,
                        num_epochs, learning_rate, weight_decay, batch_size, nn.MSELoss())

    # 绘制训练曲线
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, num_epochs + 1), train_ls)
    plt.xlabel('epoch')
    plt.ylabel('log rmse')
    plt.xlim(1, num_epochs)
    plt.yscale('log')
    plt.show()

    print(f'训练log rmse：{train_ls[-1]:.4f}')

    # 生成预测
    preds = net(test_features).detach().numpy()
    test_data['SalePrice'] = pd.Series(preds.reshape(-1))
    submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
    submission.to_csv('submission.csv', index=False)

    return submission


# ============================================
# 第八部分：主程序
# ============================================

def main() -> None:
    """
    主函数：执行完整的房价预测流程
    """
    # ------------------------------
    # 1. 配置数据集
    # ------------------------------
    DATA_HUB['kaggle_house_train'] = (
        DATA_URL + 'kaggle_house_pred_train.csv',
        '585e9cc93e70b39160e7921475f9bcd7d31219ce'
    )

    DATA_HUB['kaggle_house_test'] = (
        DATA_URL + 'kaggle_house_pred_test.csv',
        'fa19780a7b011d9b009e8bff8e99922a8ee2eb90'
    )

    # ------------------------------
    # 2. 下载并加载数据
    # ------------------------------
    print("正在加载数据...")
    train_data = pd.read_csv(download('kaggle_house_train'))
    test_data = pd.read_csv(download('kaggle_house_test'))

    print(f'训练集形状: {train_data.shape}')
    print(f'测试集形状: {test_data.shape}')

    # ------------------------------
    # 3. 数据预处理
    # ------------------------------
    train_features, test_features, train_labels = preprocess_data(train_data, test_data)

    # 获取特征数量
    features_num = train_features.shape[1]

    # ------------------------------
    # 4. 定义超参数
    # ------------------------------
    k = 5
    num_epochs = 100
    learning_rate = 0.05
    weight_decay = 0.01
    batch_size = 64

    # ------------------------------
    # 5. 执行K折交叉验证
    # ------------------------------
    print(f"\n执行{k}折交叉验证...")
    train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs,
                              learning_rate, weight_decay, batch_size, features_num)

    print(f'{k}折交叉验证结果：')
    print(f'平均训练log rmse: {train_l:.4f}')
    print(f'平均验证log rmse: {valid_l:.4f}')

    # ------------------------------
    # 6. 使用全部数据训练并生成预测
    # ------------------------------
    print("\n使用全部训练数据训练最终模型...")
    submission = train_and_pred(
            train_features, test_features, train_labels, test_data,
            num_epochs, learning_rate, weight_decay, batch_size, features_num
    )

    print(f"预测结果已保存到 submission.csv")
    print(f"提交数据形状: {submission.shape}")


if __name__ == "__main__":
    main()
