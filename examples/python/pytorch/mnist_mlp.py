import torch.nn as nn
import flexflow.torch.fx as fx

class MLP(nn.Module):
  def __init__(self):
    super().__init__()
    self.linear1 = nn.Linear(784, 512)
    self.linear2 = nn.Linear(512, 512)
    self.linear3 = nn.Linear(512, 10)
    self.relu = nn.ReLU()
    self.softmax = nn.Softmax()

  def forward(self, x):
    y = self.linear1(x)
    y = self.relu(y)
    y = self.linear2(y)
    y = self.relu(y)
    y = self.linear3(y)
    y = self.softmax(y)
    return y

model = MLP()
fx.torch_to_flexflow(model, "mlp.ff")