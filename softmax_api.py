import torch
from torch import nn
from d2l import torch as d2l
import my_train_predict as tp

if __name__ == '__main__':
    batch_size = 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

    net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28* 28 ,10)
    )

    def init_weight(m):
        if type(m) == nn.Linear:
            nn.init.normal_(m.weight,std = 0.01)

    net.apply(init_weight)

    loss = nn.CrossEntropyLoss()

    trainer = torch.optim.SGD(net.parameters(), lr = 0.05)

    tp.train_model(net,train_iter,test_iter,loss,trainer,show_plot = False)
    tp.predict(net,test_iter,6)