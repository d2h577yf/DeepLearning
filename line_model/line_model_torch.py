import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l
from torch import nn

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)

def load_array(data_arrays,batch_size,is_train=True):
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset,batch_size,shuffle=is_train)

batch_size = 10
data_iter = load_array((features, labels), batch_size)

net = nn.Sequential(nn.Linear(2,1))

# params = list(net.parameters())
# print(f"总参数数量: {len(params)}")
# for i, param in enumerate(params):
#     print(f"参数 {i}: 形状 = {param.shape}, 大小 = {param.numel()}")

net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)

# loss = nn.MSELoss()
loss = nn.SmoothL1Loss()

trainer = torch.optim.SGD(net.parameters(), lr=0.01)

num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X) ,y)
        trainer.zero_grad()
        l.backward()
        # print(f"grad weigt :{net[0].weight.grad}")
        # print(f"grad bias :{net[0].bias.grad}")
        trainer.step()
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')

w = net[0].weight.data
print('w的估计误差：', true_w - w.reshape(true_w.shape))
b = net[0].bias.data
print('b的估计误差：', true_b - b)