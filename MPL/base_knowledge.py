import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import xlabel

x = torch.arange(-8.0,8.0,0.01,requires_grad = True)
y = torch.prelu(x,torch.Tensor([0.01]))

y.sum().backward()

plt.figure(figsize=[5, 2.6])
plt.plot(x.detach(), y.detach(), label='XXX(x)')
plt.plot(x.detach(), x.grad.detach(), label='Gradient of XXX', linestyle='--')
plt.xlabel('x')
plt.ylabel('Value')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
