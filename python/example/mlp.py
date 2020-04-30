from flexflow.core import *
import numpy as np
from flexflow.keras.datasets import mnist

def top_level_task():
  alexnetconfig = NetConfig()
  ffconfig = FFConfig()
  ffconfig.parse_args()
  print("Python API batchSize(%d) workersPerNodes(%d) numNodes(%d)" %(ffconfig.get_batch_size(), ffconfig.get_workers_per_node(), ffconfig.get_num_nodes()))
  ffmodel = FFModel(ffconfig)
  
  dims1 = [ffconfig.get_batch_size(), 784]
  input1 = ffmodel.create_tensor_2d(dims1, "", DataType.DT_FLOAT);
  
  dims_label = [ffconfig.get_batch_size(), 1]
  label = ffmodel.create_tensor_2d(dims_label, "", DataType.DT_INT32);
  
  use_external = True
  if (use_external == True):
    num_samples = 60000
    
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    x_train = x_train.reshape(60000, 784)
    x_train = x_train.astype('float32')
    x_train /= 255
    y_train = y_train.astype('int32')
    print(x_train.shape[0], 'train samples')
    print(y_train.shape)
    
    dims_full_input = [num_samples, 784]
    full_input = ffmodel.create_tensor_2d(dims_full_input, "", DataType.DT_FLOAT)

    dims_full_label = [num_samples, 1]
    full_label = ffmodel.create_tensor_2d(dims_full_label, "", DataType.DT_INT32)

    full_input.attach_numpy_array(ffconfig, x_train)
    full_label.attach_numpy_array(ffconfig, y_train)
    print(y_train)

    dataloader = DataLoader2D(ffmodel, input1, label, full_input, full_label, num_samples)

    full_input.detach_numpy_array(ffconfig)
    full_label.detach_numpy_array(ffconfig)
  else:
    # Data Loader
    input1.inline_map(ffconfig)
    input1.inline_unmap(ffconfig)
    label.inline_map(ffconfig)
    label.inline_unmap(ffconfig)
  
  t2 = ffmodel.dense("dense1", input1, 512, ActiMode.AC_MODE_RELU)
  t3 = ffmodel.dense("dense1", t2, 512, ActiMode.AC_MODE_RELU)
  t4 = ffmodel.dense("dense1", t3, 10, ActiMode.AC_MODE_RELU)
  t5 = ffmodel.softmax("softmax", t4, label)

  ffoptimizer = SGDOptimizer(ffmodel, 0.01)
  ffmodel.set_sgd_optimizer(ffoptimizer)

  ffmodel.init_layers()

  epochs = ffconfig.get_epochs()

  ts_start = ffconfig.get_current_time()
  for epoch in range(0,epochs):
    dataloader.reset()
    ffmodel.reset_metrics()
    iterations = dataloader.get_num_samples() / ffconfig.get_batch_size()
    for iter in range(0, int(iterations-5)):
      dataloader.next_batch(ffmodel)
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
  print("epochs %d, ELAPSED TIME = %.4fs, THROUGHPUT = %.2f samples/s\n" %(epochs, run_time, dataloader.get_num_samples() * epochs / run_time));
 #
  dense1 = ffmodel.get_layer_by_id(0)

  dbias_tensor = dense1.get_bias_tensor()
  dbias_tensor.inline_map(ffconfig)
  dbias = dbias_tensor.get_array(ffconfig, DataType.DT_FLOAT)
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
  print("alexnet")
  top_level_task()
