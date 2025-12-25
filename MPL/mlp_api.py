import torch
from torch import nn
from d2l import torch as d2l
import my_train_predict_deprecated as tp

if __name__ == '__main__':
    net1 = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))

    net2 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256), nn.ReLU(),
            nn.Linear(256, 10)
    )

    net3 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 10)
    )

    net4 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 10)
    )

    net5 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256), nn.Sigmoid(),
            nn.Linear(256, 10)
    )

    net6 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256), nn.Tanh(),
            nn.Linear(256, 10)
    )

    net7 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256), nn.LeakyReLU(0.01),
            nn.Linear(256, 10)
    )

    net8 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256), nn.ELU(alpha = 1.0),
            nn.Linear(256, 10)
    )


    net = net2

    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.normal_(m.weight,std = 1 / 28)

    def init_weights_kai(m):
        if type(m) == nn.Linear:
            nn.init.kaiming_normal_(m.weight, mode = 'fan_out', nonlinearity = 'relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


    def init_weights_xav(m):
        if type(m) == nn.Linear:
            nn.init.xavier_normal_(m.weight, gain = 1.0)
            # gain参数：ReLU用√2，tanh用1，sigmoid用1

    net.apply(init_weights_kai)

    batch_size, lr, num_epochs = 256, 0.1, 10
    loss = nn.CrossEntropyLoss()
    trainer = torch.optim.SGD(net.parameters(), lr = lr)

    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

    tp.train_model(net,train_iter,test_iter,loss,trainer)
    tp.predict(net,test_iter,10)