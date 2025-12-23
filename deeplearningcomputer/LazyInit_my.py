import torch
from torch import nn

# 想写一个自动判断纬度的类，可是比我想的难多了
# class myadjustnet(nn.Module):
#     def __init__(self,dims):
#         super().__init__()
#         self.dims = dims
#         self.net = nn.Sequential(
#                  nn.Linear(self.dims,4),
#                  nn.ReLU(),
#                  nn.Linear(4,1),
#         )
#
#     def forward(self,X:torch.Tensor):
#         self.dims = X.shape[1]
#         return self.net(X)

def main():
    X = torch.rand(1,1)
    # net = nn.Sequential(
    #         nn.LazyLinear(4),
    #         nn.ReLU(),
    #         nn.Linear(4,1)
    # )
    # print(net[0].weight)  # 尚未初始化
    # print(net)
    #
    # print(X,"                 ",net(X))
    #
    # print(net[0].weight)  # 尚未初始化
    # print(net)
    print()

if __name__ == '__main__':
    main()