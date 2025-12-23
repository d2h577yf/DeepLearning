import torch
from torch import nn
from d2l import torch as d2l

def com_conv2d(conv2d,X:torch.Tensor):
    X = X.reshape((1,1) + X.shape)
    Y:torch.Tensor = conv2d(X)
    return Y.reshape(Y.shape[2:])

def main() -> None:
    conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=2)
    X = torch.rand(size = (8,8))
    print(com_conv2d(conv2d,X).shape)

if __name__ == '__main__':
    main()