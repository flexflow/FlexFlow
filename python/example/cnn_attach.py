from flexflow.core import *
from flexflow.keras.datasets import cifar10

def next_batch(idx, x_train, input1, ffconfig):
  start = idx*ffconfig.get_batch_size()
  x_train_batch = x_train[start:start+ffconfig.get_batch_size(), :, :, :]
  print(x_train_batch.shape)
  
  input1.inline_map(ffconfig)
  input_array = input1.get_array(ffconfig, DataType.DT_FLOAT)
  print(input_array.shape)
  for i in range(0, ffconfig.get_batch_size()):
    for j in range(0, 3):
      for k in range(0, 32):
        for l in range(0, 32):
          input_array[i][j][k][l] = x_train_batch[i][j][k][l]
  input1.inline_unmap(ffconfig)
  
def next_batch_label(idx, x_train, input1, ffconfig):
  start = idx*ffconfig.get_batch_size()
  x_train_batch = x_train[start:start+ffconfig.get_batch_size(), :]
  print(x_train_batch.shape)
  
  input1.inline_map(ffconfig)
  input_array = input1.get_array(ffconfig, DataType.DT_INT32)
  print(input_array.shape)
  for i in range(0, ffconfig.get_batch_size()):
    for j in range(0, 1):
      input_array[i][j] = x_train_batch[i][j]
  input1.inline_unmap(ffconfig)

def top_level_task():
  ffconfig = FFConfig()
  alexnetconfig = NetConfig()
  print(alexnetconfig.dataset_path)
  ffconfig.parse_args()
  print("Python API batchSize(%d) workersPerNodes(%d) numNodes(%d)" %(ffconfig.get_batch_size(), ffconfig.get_workers_per_node(), ffconfig.get_num_nodes()))
  ffmodel = FFModel(ffconfig)
  
  dims_input = [ffconfig.get_batch_size(), 3, 32, 32]
  input = ffmodel.create_tensor(dims_input, "", DataType.DT_FLOAT)

  dims_label = [ffconfig.get_batch_size(), 1]
  label = ffmodel.create_tensor(dims_label, "", DataType.DT_INT32)
  
  num_samples = 10000
  
  (x_train, y_train), (x_test, y_test) = cifar10.load_data(num_samples)
  
  x_train = x_train.astype('float32')
  x_train /= 255
  
  # x_train_t = x_train.transpose(3, 2, 1, 0)
  #
  # x_train = np.zeros((32,32,3,num_samples), dtype=np.float32)
  #
  # for i in range(0, num_samples):
  #   for j in range(0, 3):
  #     for k in range(0, 32):
  #       for l in range(0, 32):
  #         x_train[l][k][j][i] = x_train_t[l][k][j][i]

  full_input_array = x_train
  print(full_input_array.__array_interface__["strides"])

  
  y_train = y_train.astype('int32')

  full_label_array = y_train
 
  print(full_input_array.__array_interface__["strides"])
  print(full_input_array.shape, full_label_array.shape)
  print(full_label_array.__array_interface__["strides"])
  
  next_batch(0, x_train, input, ffconfig)
  next_batch_label(0, y_train, label, ffconfig)

  t = ffmodel.conv2d("conv1", input, 32, 3, 3, 1, 1, 1, 1, ActiMode.AC_MODE_RELU)
  t = ffmodel.conv2d("conv2", t, 32, 3, 3, 1, 1, 1, 1, ActiMode.AC_MODE_RELU)
  t = ffmodel.pool2d("pool1", t, 2, 2, 2, 2, 0, 0,)
  t = ffmodel.conv2d("conv3", t, 64, 3, 3, 1, 1, 1, 1, ActiMode.AC_MODE_RELU)
  t = ffmodel.conv2d("conv4", t, 64, 3, 3, 1, 1, 1, 1, ActiMode.AC_MODE_RELU)
  t = ffmodel.pool2d("pool2", t, 2, 2, 2, 2, 0, 0)
  t = ffmodel.flat("flat", t);
  t = ffmodel.dense("lienar1", t, 512, ActiMode.AC_MODE_RELU)
  t = ffmodel.dense("lienar1", t, 10)
  t = ffmodel.softmax("softmax", t, label)

  ffoptimizer = SGDOptimizer(ffmodel, 0.01)
  ffmodel.set_sgd_optimizer(ffoptimizer)

  ffmodel.init_layers()



  epochs = ffconfig.get_epochs()
  #epochs = 10

  ts_start = ffconfig.get_current_time()
  for epoch in range(0,epochs):
    ffmodel.reset_metrics()
    iterations = int(num_samples / ffconfig.get_batch_size())
    print(iterations, num_samples)
    ct = 0
    for iter in range(0, int(iterations)):
      next_batch(ct, x_train, input, ffconfig)
      next_batch_label(ct, y_train, label, ffconfig)   
      ct += 1
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
  #ffmodel.print_layers(13)

  conv_2d1 = ffmodel.get_layer_by_id(0)
  cbias_tensor = conv_2d1.get_input_tensor()
  #cbias_tensor = conv_2d1.get_output_tensor()
  cbias_tensor.inline_map(ffconfig)
  cbias = cbias_tensor.get_flat_array(ffconfig, DataType.DT_FLOAT)
  print(cbias.shape)
  print(cbias)
  cbias_tensor.inline_unmap(ffconfig)

  label.inline_map(ffconfig)
  label_array = label.get_flat_array(ffconfig, DataType.DT_INT32)
  print(label_array.shape)
  # print(cbias)
  print(label_array)
  label.inline_unmap(ffconfig)


if __name__ == "__main__":
  print("alexnet")
  top_level_task()
