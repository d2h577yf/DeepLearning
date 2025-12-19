import torch
from torch import nn
from torch.nn import functional as F

# class MLP(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.hidden = nn.Linear(20,256)
#         self.out = nn.Linear(256,10)
#
#     def forward(self,X):
#         return self.out(F.relu(self.hidden(X)))
#
# class MySequential(nn.Module):
#     def __init__(self,*args):
#         super().__init__()
#         for idx, model in enumerate(args):
#             self._modules[str(idx)] = model
#
#     def forward(self,X):
#         for block in self._modules.values():
#             X = block(X)
#         return X

class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.rand_weight = torch.rand((20,20),requires_grad = False)
        self.liner = nn.Linear(20,20)

    def forward(self,X):
        X = self.liner(X)
        X = F.relu((X @ self.rand_weight) + 1)
        x = self.liner(X)

        while X.abs().sum() > 1:
            X /= 2
        return X.sum()

class NestMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(20,64),nn.ReLU(),
                                 nn.Linear(64,32),nn.ReLU())
        self.linear = nn.Linear(32,16)

    def forward(self,X):
        return self.linear(self.net(X))

def main() -> None:
    X = torch.rand(2,20)
    # net = MLP()
    # net = MySequential(nn.Linear(20, 256),
    #                    nn.ReLU(),
    #                    nn.Linear(256, 10))
    # net = FixedHiddenMLP()
    net = nn.Sequential(NestMLP(),nn.Linear(16,20),FixedHiddenMLP())
    print(f'X:{X}\nnet(X):{net(X)}')


if __name__ == '__main__':
    main()