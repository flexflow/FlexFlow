import torch.nn as nn
import torch
from flexflow.torch.model import PyTorchModel

class CNN(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(3, 32, 3, 1)
    self.conv2 = nn.Conv2d(32, 32, 3, 1)
    self.pool1 = nn.MaxPool2d(2, 2)
    self.conv3 = nn.Conv2d(32, 64, 3, 1)
    self.conv4 = nn.Conv2d(64, 64, 3, 1)
    self.pool2 = nn.MaxPool2d(2, 2)
    self.flat1 = nn.Flatten()
    self.linear1 = nn.Linear(512, 512)
    self.linear2 = nn.Linear(512, 10)
    self.relu = nn.ReLU()

  def forward(self, input1, input2):
    y1 = self.conv1(input1)
    y1 = self.relu(y1)
    y2 = self.conv1(input2)
    y2 = self.relu(y2)
    y = torch.cat((y1, y2), 1)
    (y1, y2) = torch.split(y, 1) 
    y = torch.cat((y1, y2), 1)
    y = self.conv2(y)
    y = self.relu(y)
    y = self.pool1(y)
    y = self.conv3(y)
    y = self.relu(y)
    y = self.conv4(y)
    y = self.relu(y)
    y = self.pool2(y)
    y = self.flat1(y)
    y = self.linear1(y)
    y = self.relu(y)
    yo = self.linear2(y)
    return (yo, y)

model = CNN()
ff_torch_model = PyTorchModel(model)
ff_torch_model.torch_to_file("cnn.ff")

