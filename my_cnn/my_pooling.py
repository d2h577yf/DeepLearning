import torch
from torch import nn
from d2l import torch as d2l

# def pool2d(X:torch.Tensor,pool_size,mode = 'max'):
#     p_h,p_w = pool_size
#     Y = torch.zeros((X.shape[0] - p_h + 1,X.shape[1] - p_w + 1))
#     for i in range(Y.shape[0]):
#        for j in range(Y.shape[1]):
#            if mode == 'max':
#                Y[i,j] = X[i : i + p_h,j:j + p_w].max()
#            if mode == 'avg':
#                Y[i,j] = X[i : i + p_h,j:j + p_w].mean()
#     return Y

def main() -> None:
    # X = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
    # print(pool2d(X, (2, 2)))
    
    X = torch.arange(16, dtype = torch.float32).reshape((1, 1, 4, 4))
    # print(X)
    # 默认情况下步幅与汇聚窗口的大小相同,所以这里只会输出一个数
    # pool2d_1 = nn.MaxPool2d(3)
    # print(pool2d_1(X))
    # pool2d_2 = nn.MaxPool2d(3,padding = 1,stride = 2)
    # print(pool2d_2(X))
    # pool2d_3 = nn.MaxPool2d((2, 3), stride = (2, 3), padding = (0, 1))
    # print(pool2d_3(X))
    X_stack = torch.cat((X, X + 1), 1)
    pool2d_4 = nn.MaxPool2d(3, padding = 1, stride = 2)
    print(pool2d_4(X_stack))

if __name__ == '__main__':
    main()