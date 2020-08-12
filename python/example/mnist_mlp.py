from flexflow.core import *
import numpy as np
from flexflow.keras.datasets import mnist

from accuracy import ModelAccuracy

def top_level_task():
  ffconfig = FFConfig()
  ffconfig.parse_args()
  print("Python API batchSize(%d) workersPerNodes(%d) numNodes(%d)" %(ffconfig.get_batch_size(), ffconfig.get_workers_per_node(), ffconfig.get_num_nodes()))
  ffmodel = FFModel(ffconfig)
  
  dims1 = [ffconfig.get_batch_size(), 784]
  input1 = ffmodel.create_tensor(dims1, "", DataType.DT_FLOAT);
  
  # dims_label = [ffconfig.get_batch_size(), 1]
  # label = ffmodel.create_tensor(dims_label, "", DataType.DT_INT32);
  
  num_samples = 60000
  
  kernel_init = UniformInitializer(12, -1, 1)
  t = ffmodel.dense(input1, 512, ActiMode.AC_MODE_RELU, kernel_initializer=kernel_init)
  t = ffmodel.dense(t, 512, ActiMode.AC_MODE_RELU)
  t = ffmodel.dense(t, 10)
  
  t = ffmodel.softmax(t)

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
  print(x_train.shape[0], 'train samples')
  print(y_train.shape)
  
  dims_full_input = [num_samples, 784]
  full_input = ffmodel.create_tensor(dims_full_input, "", DataType.DT_FLOAT)

  dims_full_label = [num_samples, 1]
  full_label = ffmodel.create_tensor(dims_full_label, "", DataType.DT_INT32)

  full_input.attach_numpy_array(ffconfig, x_train)
  full_label.attach_numpy_array(ffconfig, y_train)
  print(y_train)

  #dataloader = DataLoader2D(ffmodel, input1, label, full_input, full_label, num_samples)
  dataloader_input = SingleDataLoader(ffmodel, input1, full_input, num_samples, DataType.DT_FLOAT)
  dataloader_label = SingleDataLoader(ffmodel, label, full_label, num_samples, DataType.DT_INT32)

  full_input.detach_numpy_array(ffconfig)
  full_label.detach_numpy_array(ffconfig)

  ffmodel.init_layers()

  epochs = ffconfig.get_epochs()

  ts_start = ffconfig.get_current_time()
  for epoch in range(0,epochs):
    dataloader_input.reset()
    dataloader_label.reset()
    # dataloader.reset()
    ffmodel.reset_metrics()
    iterations = num_samples / ffconfig.get_batch_size()
    for iter in range(0, int(iterations)):
      dataloader_input.next_batch(ffmodel)
      dataloader_label.next_batch(ffmodel)
      #dataloader.next_batch(ffmodel)
      if (epoch > 0):
        ffconfig.begin_trace(111)
      ffmodel.forward()
      ffmodel.zero_gradients()
      ffmodel.backward()
      ffmodel.update()
      if (epoch > 0):
        ffconfig.end_trace(111)

  ts_end = ffconfig.get_current_time()
  run_time = 1e-6 * (ts_end - ts_start);
  print("epochs %d, ELAPSED TIME = %.4fs, THROUGHPUT = %.2f samples/s\n" %(epochs, run_time, num_samples * epochs / run_time));
  
  perf_metrics = ffmodel.get_perf_metrics()
  accuracy = perf_metrics.get_accuracy()
  if accuracy < ModelAccuracy.MNIST_MLP.value:
    assert 0, 'Check Accuracy'

  dense1 = ffmodel.get_layer_by_id(0)

  dbias_tensor = label#dense1.get_bias_tensor()
  dbias_tensor.inline_map(ffconfig)
  dbias = dbias_tensor.get_array(ffconfig, DataType.DT_INT32)
  print(dbias.shape)
  print(dbias)
  dbias_tensor.inline_unmap(ffconfig)

  # dweight_tensor = dense1.get_output_tensor()
  # dweight_tensor.inline_map(ffconfig)
  # dweight = dweight_tensor.get_array(ffconfig, DataType.DT_FLOAT)
  # print(dweight.shape)
  # print(dweight)
  # dweight_tensor.inline_unmap(ffconfig)
  
  
if __name__ == "__main__":
  print("mnist mlp")
  top_level_task()
