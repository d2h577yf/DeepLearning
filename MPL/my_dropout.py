import torch
from torch import nn
from d2l import torch as d2l

dropout1,dropout2 = 0.2,0.5

def dropout_layer(X, dropout):
    assert 0 <= dropout <= 1
    # 在本情况中，所有元素都被丢弃
    if dropout == 1:
        return torch.zeros_like(X)
    # 在本情况中，所有元素都被保留
    if dropout == 0:
        return X
    mask = (torch.rand(X.shape) > dropout).float()
    return mask * X / (1.0 - dropout)

class Net(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_hiddens1, num_hiddens2,
                 is_training = True):
        super(Net, self).__init__()
        self.num_inputs = num_inputs
        self.training = is_training
        self.lin1 = nn.Linear(num_inputs, num_hiddens1)
        self.lin2 = nn.Linear(num_hiddens1, num_hiddens2)
        self.lin3 = nn.Linear(num_hiddens2, num_outputs)
        self.relu = nn.ReLU()

    def forward(self, X):
        H1 = self.relu(self.lin1(X.reshape((-1, self.num_inputs))))
        if self.training:
            H1 = dropout_layer(H1, dropout1)
        H2 = self.relu(self.lin2(H1))
        if self.training:
            H2 = dropout_layer(H2, dropout2)
        out = self.lin3(H2)
        return out

if __name__ == '__main__':
    num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256

    num_epochs , lr , batch_size = 10,0.1,256

    # net = Net(num_inputs,num_outputs,num_hiddens1,num_hiddens2)
    net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_inputs,num_hiddens1),
            nn.ReLU(),
            nn.Dropout(dropout1),

            nn.Linear(num_hiddens1,num_hiddens2),
            nn.ReLU(),
            nn.Dropout(dropout2),

            nn.Linear(num_hiddens2,num_outputs)
    )

    def init_weight(m):
        if type(m) == nn.Linear:
            nn.init.normal_(m.weight,std = 0.01)

    net.apply(init_weight)

    loss = nn.CrossEntropyLoss()

    trainer = torch.optim.SGD(net.parameters(),lr=lr,weight_decay = 0.01)

    train_iter , test_iter = d2l.load_data_fashion_mnist(batch_size)

    # tp.train_model(net,train_iter,test_iter,loss,trainer,num_epochs)
    # tp.predict(net,test_iter,num_epochs)

    print("开始训练。。。。")
    train_loss_history = []
    train_acc_history = []
    test_acc_history = []

    for epoch in range(num_epochs):
        net.train()

        total_train_loss = 0.0
        total_train_correct = 0
        total_train_samples = 0

        for X,y in train_iter:
            y_hat = net(X)
            l = loss(y_hat,y)

            trainer.zero_grad()
            l.backward()
            trainer.step()

            batch_loss = l.item()
            batch_correct = (y_hat.argmax(dim = 1) == y).sum().item()
            bach_size = y.shape[0]

            total_train_loss += batch_loss * batch_size
            total_train_correct += batch_correct
            total_train_samples += batch_size

        avg_train_loss = total_train_loss / total_train_samples
        train_acc = total_train_correct / total_train_samples
        train_loss_history.append(avg_train_loss)
        train_acc_history.append(train_acc)

        net.eval()

        total_test_correct = 0
        total_test_samples = 0

        with torch.no_grad():
            for X, y in test_iter:
                y_hat = net(X)
                test_correct = (y_hat.argmax(dim=1) == y).sum().item()

                total_test_correct += test_correct
                total_test_samples += y.shape[0]

        test_acc = total_test_correct / total_test_samples
        test_acc_history.append(test_acc)

        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  训练损失: {avg_train_loss:.4f}, 训练准确率: {train_acc:.3f}")
        print(f"  测试准确率: {test_acc:.3f}")

    print("\n训练完成！")
    print(f"最终测试准确率: {test_acc_history[-1]:.3f}")

    import matplotlib.pyplot as plt

    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Songti SC', 'STHeiti']
    plt.rcParams['axes.unicode_minus'] = False

    plt.figure(figsize = (12, 4))

    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), train_loss_history, 'b-', label = '训练损失')
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.title('训练损失变化曲线')
    plt.grid(True)

    # 准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), train_acc_history, 'r-', label = '训练准确率')
    plt.plot(range(1, num_epochs + 1), test_acc_history, 'g-', label = '测试准确率')
    plt.xlabel('Epoch')
    plt.ylabel('准确率')
    plt.title('训练与测试准确率变化曲线')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()