#./flexflow_python $FF_HOME/bootcamp_demo/torch_alexnet_cifar10.py -ll:py 1 -ll:gpu 1 -ll:fsize 2048 -ll:zsize 12192

# https://github.com/pytorch/vision/blob/master/torchvision/models/alexnet.py

import torch.nn as nn
import torch
import flexflow.torch.fx as fx
import torchvision.models as models

class AlexNet(nn.Module):
  def __init__(self, num_classes: int = 1000) -> None:
    super(AlexNet, self).__init__()
    self.features = nn.Sequential(
      nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=3, stride=2),
      nn.Conv2d(64, 192, kernel_size=5, padding=2),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=3, stride=2),
      nn.Conv2d(192, 384, kernel_size=3, padding=1),
      nn.ReLU(inplace=True),
      nn.Conv2d(384, 256, kernel_size=3, padding=1),
      nn.ReLU(inplace=True),
      nn.Conv2d(256, 256, kernel_size=3, padding=1),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=3, stride=2),
    )
    self.classifier = nn.Sequential(
      nn.Linear(256 * 6 * 6, 4096),
      nn.ReLU(inplace=True),
      nn.Linear(4096, 4096),
      nn.ReLU(inplace=True),
      nn.Linear(4096, num_classes),
      nn.Softmax(),
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.features(x)
    x = torch.flatten(x, 1)
    x = self.classifier(x)
    return x

model = AlexNet(num_classes=10)
fx.torch_to_flexflow(model, "alexnet.ff")