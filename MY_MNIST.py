import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l
import matplotlib.pyplot as plt
import numpy as np
import os


def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols*scale, num_rows*scale))
    for i, ax in enumerate(axes.reshape(-1)):
        ax.imshow(imgs[i].numpy() if hasattr(imgs[i], 'numpy') else imgs[i])
        ax.set_axis_off()
        if titles: ax.set_title(titles[i])
    plt.show()
    return axes

def get_dataloader_workers():  #@save
    return 5

def load_data_fashion_mnist(batch_size, resize=None):  #@save
    """下载Fashion-MNIST数据集，然后将其加载到内存中"""
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root="../data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="../data", train=False, transform=trans, download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                            num_workers=get_dataloader_workers(),persistent_workers=True,prefetch_factor=4),
            data.DataLoader(mnist_test, batch_size, shuffle=False,
                            num_workers=get_dataloader_workers(),persistent_workers=True,prefetch_factor=4))

# trans = transforms.ToTensor()
# mnist_train = torchvision.datasets.FashionMNIST(
#     root="../data", train=True, transform=trans, download=True)
# mnist_test = torchvision.datasets.FashionMNIST(
#     root="../data", train=False, transform=trans, download=True)

# print(len(mnist_train), len(mnist_test))
# print(mnist_train[0][0].shape)

# X, y = next(iter(data.DataLoader(mnist_train, batch_size=18)))
# show_images(X.reshape(18, 28, 28), 2, 9, titles=get_fashion_mnist_labels(y));

# train_iter = data.DataLoader(mnist_train,batch_size=256,shuffle=True,num_workers=get_dataloader_workers())
#
# timer = d2l.Timer()
# for X, y in train_iter:
#     continue
# f'{timer.stop():.2f} sec'

# print("worker is here")

if __name__ == '__main__':
    batch_size = 256
    train_iter, test_iter = load_data_fashion_mnist(batch_size, resize=64)

    # 测试数据加载速度
    timer = d2l.Timer()
    for X, y in train_iter:
        continue
    print(f'{timer.stop():.2f} sec')
#
#
# def list_all_torchvision_datasets():
#     """列出所有torchvision数据集"""
#     all_datasets = []
#
#     # 获取torchvision.datasets模块的所有属性
#     for attr_name in dir(torchvision.datasets):
#         attr = getattr(torchvision.datasets, attr_name)
#
#         # 检查是否是数据集类（排除模块、函数等）
#         if isinstance(attr, type) and hasattr(attr, '__name__'):
#             if 'Dataset' in attr.__name__ or 'Data' in attr.__name__:
#                 all_datasets.append(attr.__name__)
#
#     return sorted(all_datasets)
#
#
# print("Torchvision所有数据集类:")
# datasets = list_all_torchvision_datasets()
# for i, dataset in enumerate(datasets, 1):
#     print(f"{i:3d}. {dataset}")