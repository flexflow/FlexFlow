import torch.nn as nn
from flexflow.core import *
import numpy as np
from flexflow.keras.datasets import mnist
from flexflow.torch.model import PyTorchModel

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

def top_level_task():
  ffconfig = FFConfig()
  print("Python API batchSize(%d) workersPerNodes(%d) numNodes(%d)" %(ffconfig.batch_size, ffconfig.workers_per_node, ffconfig.num_nodes))
  ffmodel = FFModel(ffconfig)

  dims = [ffconfig.batch_size, 784]
  input_tensor = ffmodel.create_tensor(dims, DataType.DT_FLOAT);

  num_samples = 60000
  
  model = MLP()
  
  ff_torch_model = PyTorchModel(model)
  output_tensors = ff_torch_model.torch_to_ff(ffmodel, [input_tensor])

  ffoptimizer = SGDOptimizer(ffmodel, 0.01)
  ffmodel.optimizer = ffoptimizer
  ffmodel.compile(loss_type=LossType.LOSS_SPARSE_CATEGORICAL_CROSSENTROPY, metrics=[MetricsType.METRICS_ACCURACY, MetricsType.METRICS_SPARSE_CATEGORICAL_CROSSENTROPY])
  label_tensor = ffmodel.label_tensor

  (x_train, y_train), (x_test, y_test) = mnist.load_data()

  print(x_train.shape)
  x_train = x_train.reshape(60000, 784)
  x_train = x_train.astype('float32')
  x_train /= 255
  y_train = y_train.astype('int32')
  y_train = np.reshape(y_train, (len(y_train), 1))

  dataloader_input = ffmodel.create_data_loader(input_tensor, x_train)
  dataloader_label = ffmodel.create_data_loader(label_tensor, y_train)

  ffmodel.init_layers()

  epochs = ffconfig.epochs

  ts_start = ffconfig.get_current_time()
  
  ffmodel.fit(x=dataloader_input, y=dataloader_label, epochs=epochs)

  ts_end = ffconfig.get_current_time()
  run_time = 1e-6 * (ts_end - ts_start);
  print("epochs %d, ELAPSED TIME = %.4fs, THROUGHPUT = %.2f samples/s\n" %(epochs, run_time, num_samples * epochs / run_time));

  # perf_metrics = ffmodel.get_perf_metrics()
  # accuracy = perf_metrics.get_accuracy()
  # if accuracy < ModelAccuracy.MNIST_MLP.value:
  #   assert 0, 'Check Accuracy'

if __name__ == "__main__":
  print("mnist mlp")
  top_level_task()
