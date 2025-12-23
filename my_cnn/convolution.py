import torch
from torch import nn, einsum
from d2l import torch as d2l

def corr2d(x:torch.Tensor,k:torch.Tensor) -> torch.Tensor:
    row,col = k.shape
    y = torch.zeros(x.shape[0] - row + 1,x.shape[1] - col + 1)
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            y[i,j] = (x[i : i + row, j : j + col] * k).sum()
    return y
#
# class Conv2D(nn.Module):
#     def __init__(self,k_size):
#         super().__init__()
#         self.weight = nn.Parameter(torch.rand(k_size))
#         self.bias = nn.Parameter(torch.zeros(1))
#
#     def forward(self,x):
#         return corr2d(x,self.weight) + self.bias

def main():
    # x = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
    # k = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
    # y = corr2d(x, k)
    # print(y)
    x = torch.tensor([
        [0,0,0,1,0,0,0],
        [0,0,1,0,1,0,0],
        [0,1,0,0,0,1,0]
    ])
    #
    k = torch.tensor([[1.0, -1.0]])
    y1 = corr2d(x,k)
    print(y1)
    # # y2 = corr2d(x.t(),k)
    # # print(f"y1 \n:{y1} \n y2 \n :{y2} ")
    #
    # conv2d = nn.Conv2d(1,1,kernel_size = (1,2),bias = False)
    #
    # x = x.reshape((1,1,*x.shape))
    # y = y1.reshape((1,1,*y1.shape))
    # lr = 3e-2
    #
    # for epoch in range(11):
    #     y_hat = conv2d(x)
    #     l = (y_hat - y) ** 2
    #     conv2d.zero_grad()
    #     l.sum().backward()
    #
    #     conv2d.weight.data[:] -= lr * conv2d.weight.grad
    #     if(epoch + 1) % 2 == 0:
    #         print(f'epoch{epoch + 1},loss {l.sum():.3f}')
    #
    # print(conv2d.weight.data)



if __name__ == '__main__':
    main()