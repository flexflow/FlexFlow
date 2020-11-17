from flexflow.core import *
import numpy as np
from flexflow.keras.datasets import mnist
from flexflow.torch.model import PyTorchModel

#from accuracy import ModelAccuracy

def top_level_task():
  ffconfig = FFConfig()
  ffconfig.parse_args()
  print("Python API batchSize(%d) workersPerNodes(%d) numNodes(%d)" %(ffconfig.get_batch_size(), ffconfig.get_workers_per_node(), ffconfig.get_num_nodes()))
  ffmodel = FFModel(ffconfig)

  dims = [ffconfig.get_batch_size(), 784]
  input_tensor = ffmodel.create_tensor(dims, DataType.DT_FLOAT);

  num_samples = 60000

  torch_model = PyTorchModel("mlp.ff")
  
  output_tensors = torch_model.apply(ffmodel, [input_tensor])

  ffoptimizer = SGDOptimizer(ffmodel, 0.01)
  ffmodel.set_sgd_optimizer(ffoptimizer)
  ffmodel.compile(loss_type=LossType.LOSS_SPARSE_CATEGORICAL_CROSSENTROPY, metrics=[MetricsType.METRICS_ACCURACY, MetricsType.METRICS_SPARSE_CATEGORICAL_CROSSENTROPY])
  label = ffmodel.get_label_tensor()

  (x_train, y_train), (x_test, y_test) = mnist.load_data()

  print(x_train.shape)
  x_train = x_train.reshape(60000, 784)
  x_train = x_train.astype('float32')
  x_train /= 255
  y_train = y_train.astype('int32')
  y_train = np.reshape(y_train, (len(y_train), 1))

  dims_full_input = [num_samples, 784]
  full_input = ffmodel.create_tensor(dims_full_input, DataType.DT_FLOAT)

  dims_full_label = [num_samples, 1]
  full_label = ffmodel.create_tensor(dims_full_label, DataType.DT_INT32)

  full_input.attach_numpy_array(ffconfig, x_train)
  full_label.attach_numpy_array(ffconfig, y_train)

  dataloader_input = SingleDataLoader(ffmodel, input_tensor, full_input, num_samples, DataType.DT_FLOAT)
  dataloader_label = SingleDataLoader(ffmodel, label, full_label, num_samples, DataType.DT_INT32)

  full_input.detach_numpy_array(ffconfig)
  full_label.detach_numpy_array(ffconfig)

  ffmodel.init_layers()

  epochs = ffconfig.get_epochs()

  ts_start = ffconfig.get_current_time()
  
  ffmodel.train((dataloader_input, dataloader_label), epochs)

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
