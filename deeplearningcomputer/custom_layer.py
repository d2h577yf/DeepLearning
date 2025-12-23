import torch
from torch import nn

class MyTensorReductionLayerWithBias(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.weight = nn.Parameter(torch.zeros(output_dim, input_dim, input_dim))

        self.bias = nn.Parameter(torch.zeros(output_dim))

        nn.init.xavier_normal_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, X):
        y = torch.einsum('bi,kij,bj->bk', X, self.weight, X)
        y += self.bias
        return y


def main():
    input_dim,output_dim = 5,3
    net = MyTensorReductionLayerWithBias(input_dim,output_dim)
    x = torch.rand(2,input_dim)
    y = net(x)
    # print(x,"\n",y)
    # torch.save(y,"y_try")
    #
    # y2 = torch.load("y_try")
    # print(y2)
    # torch.save(net.state_dict(),'net state')
    # net2 = MyTensorReductionLayerWithBias(input_dim,output_dim)
    # net2.load_state_dict(torch.load("net state"))
    # print(net2(x))

if __name__ == '__main__':
    main()