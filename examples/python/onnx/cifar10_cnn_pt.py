import onnx
import torch
import torch.nn as nn
from torch.onnx import TrainingMode

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2))
        self.classifier = nn.Sequential(
            nn.Linear(4096, 512), 
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.ReLU(),
            nn.Softmax())

    def forward(self, x):
        x = self.features(x)
        print(x.shape)
        x = torch.flatten(x, 1)
        print(x.shape)
        x = self.classifier(x)
        return x

input = torch.randn(64, 3, 32, 32)
model = CNN()
torch.onnx.export(model, (input), "cifar10_cnn.onnx", export_params=False, training=TrainingMode.TRAINING)
