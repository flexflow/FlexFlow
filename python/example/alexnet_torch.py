import flexflow.torch.nn as nn

from flexflow.core import *

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
    self.flat = nn.Flatten()
    self.linear_1 = nn.Linear(256 * 6 * 6, 4096)
    self.linear_2 = nn.Linear(4096, 4096)
    self.linear_3 = nn.Linear(4096, 1000)
    self.test = "test"
  
  def forward(self, x):
    x = self.conv2_1(x)
    x = self.maxpool2d_1(x)
    x = self.conv2_2(x)
    x = self.maxpool2d_2(x)
    x = self.conv2_3(x)
    x = self.conv2_4(x)
    x = self.conv2_5(x)
    x = self.maxpool2d_3(x)
    x = self.flat(x)
    x = self.linear_1(x)
    x = self.linear_2(x)
    x = self.linear_3(x)
    return x
  
def top_level_task():
  model = AlexNet()
  
  dims = [model.ffconfig.get_batch_size(), 3, 229, 229]
  input = model.ffmodel.create_tensor(dims, "", DataType.DT_FLOAT);
  
  dims_label = [model.ffconfig.get_batch_size(), 1]
  label = model.ffmodel.create_tensor(dims_label, "", DataType.DT_INT32);
  dataloader = DataLoader(model.ffmodel, input, label, 1)
  
  t = model.init_inout(input)
  model.ffmodel.init_layers()
  #print(model.__dict__)
  # x.inline_map(model.ffconfig)
  # x_array = x.get_array(model.ffconfig, DataType.DT_FLOAT)
  # print(x_array.shape)
  # #print(output_array11)
  # x.inline_unmap(model.ffconfig)
  output_tensor = model.ffmodel.softmax("softmax", t, label)
  softmax = model.ffmodel.get_layer_by_id(12)
  softmax.init(model.ffmodel)

  epochs = model.ffconfig.get_epochs()

  ts_start = model.ffconfig.get_current_time()
  for epoch in range(0,epochs):
   model.ffmodel.reset_metrics()
   iterations = 8192 / model.ffconfig.get_batch_size()
   for iter in range(0, int(iterations)):
     if (epoch > 0):
       model.ffconfig.begin_trace(111)
     #model.ffmodel.forward()
     model(input)
     softmax.forward(model.ffmodel)
     #t = softmax.get_output_tensor()
     # t.inline_map(model.ffconfig)
     # input1_array = t.get_array(model.ffconfig, DataType.DT_FLOAT)
     # print(input1_array.shape)
     # t.inline_unmap(model.ffconfig)
     model.ffmodel.zero_gradients()
     model.ffmodel.backward()
     model.ffmodel.update()
     if (epoch > 0):
       model.ffconfig.end_trace(111)

  ts_end = model.ffconfig.get_current_time()
  run_time = 1e-6 * (ts_end - ts_start);
  print("epochs %d, ELAPSED TIME = %.4fs, THROUGHPUT = %.2f samples/s\n" %(epochs, run_time, 8192 * epochs / run_time));

if __name__ == "__main__":
  print("alexnet torch")
  top_level_task()