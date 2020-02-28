import flexflow.torch.nn as nn

class AlexNet(nn.Module):
  def __init__(self):
    super(AlexNet, self).__init__()
    self.conv2_1 = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)
    self.maxpool2d_1 = nn.MaxPool2d(kernel_size=3, stride=2)
    self.linear_1 = nn.Linear(256 * 6 * 6, 4096)
  
def top_level_task():
  model = AlexNet()

if __name__ == "__main__":
  print("alexnet torch")
  top_level_task()