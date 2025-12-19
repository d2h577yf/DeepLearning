import torch
from torch import nn

def main() -> None:
    net = nn.Sequential(nn.Linear(4, 8),
                        nn.ReLU(),
                        nn.Linear(8, 1))
    X = torch.rand(size = (2, 4))
    # print(net[2].state_dict())
    # print(type(net[2].bias))
    # print(net[2].bias)
    # print(net[2].bias.data)
    # print(*[(name, param.shape) for name, param in net[0].named_parameters()])
    # print(*[(name, param.shape) for name, param in net.named_parameters()])
    # print(net.state_dict()['2.bias'].data)

    # block1 = nn.Sequential(
    #         nn.Linear(4,8),
    #         nn.ReLU(),
    #         nn.Linear(8,4),
    #         nn.ReLU(),
    # )
    #
    # block2 = nn.Sequential()
    # for i in range(4):
    #     block2.add_module(f'block{i}',block1)
    #
    # rgnet = nn.Sequential(block2,nn.Linear(4,1))
    # print(rgnet(X))
    # print(rgnet)
    # print(rgnet[0][1][0].bias.data)

    def my_init(m):
        if type(m) == nn.Linear:
            print("Init", *[(name, param.shape)
                            for name, param in m.named_parameters()][0])
            nn.init.uniform_(m.weight, -10, 10)
            m.weight.data *= m.weight.data.abs() >= 5 # False = 0; True = 1

    net.apply(my_init)
    print(net[0].weight[:2])
if __name__ == '__main__':
    main()