import torch
import matplotlib.pyplot as plt
import scipy.constants as constants
import numpy as np

x = torch.linspace(-2 * constants.pi, 2 * constants.pi, 100,requires_grad=True)

y = torch.sin(x)
y.sum().backward()

plt.figure(figsize=(10,10))
plt.plot(x.detach().numpy(), x.grad.detach().numpy(),'r-*',label='dy/dx')
plt.plot(x.detach().numpy(), np.cos(x.detach().numpy()),'b--H',label='cosx')
plt.legend()
plt.grid(True)
plt.show()
