import flexflow.torch.nn as nn

class AlexNet(nn.Module):
  def __init__(self):
    super(AlexNet, self).__init__()
    self.conv2_1 = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)
    self.maxpool2d_1 = nn.MaxPool2d(kernel_size=3, stride=2)
    self.conv2_2 = nn.Conv2d(64, 192, kernel_size=5, stride=1, padding=2)
    self.maxpool2d_2 = nn.MaxPool2d(kernel_size=3, stride=2)
    self.conv2_3 = nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1)
    self.conv2_4 = nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1)
    self.conv2_5 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
    self.maxpool2d_3 = nn.MaxPool2d(kernel_size=3, stride=2)
    #self.flat = nn.Flatten()
    self.test = "test"
    #self.linear_1 = nn.Linear(256 * 6 * 6, 4096)
  
  def forward(self, x):
    x = self.conv2_1(x)
    x = self.maxpool2d_1(x)
    x = self.conv2_2(x)
    x = self.maxpool2d_2(x)
    x = self.conv2_3(x)
    x = self.conv2_4(x)
    x = self.conv2_5(x)
    x = self.maxpool2d_3(x)
    return x
  
def top_level_task():
  model = AlexNet()
  x = model(7)
  print(model.__dict__)

if __name__ == "__main__":
  print("alexnet torch")
  top_level_task()