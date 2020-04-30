from flexflow.core import *
from flexflow.keras.datasets import cifar10

def top_level_task():
  ffconfig = FFConfig()
  alexnetconfig = NetConfig()
  print(alexnetconfig.dataset_path)
  ffconfig.parse_args()
  print("Python API batchSize(%d) workersPerNodes(%d) numNodes(%d)" %(ffconfig.get_batch_size(), ffconfig.get_workers_per_node(), ffconfig.get_num_nodes()))
  ffmodel = FFModel(ffconfig)
  
  dims_input = [ffconfig.get_batch_size(), 3, 32, 32]
  #print(dims)
  input = ffmodel.create_tensor_4d(dims_input, "", DataType.DT_FLOAT)

  dims_label = [ffconfig.get_batch_size(), 1]
  #print(dims)
  label = ffmodel.create_tensor_2d(dims_label, "", DataType.DT_INT32)
  
  use_external = True
  if (use_external == True):
    num_samples = 10000
    
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32')
    x_train /= 255
    x_train = x_train.transpose(2, 3, 1, 0)
    y_train = y_train.astype('int32')
    y_train = y_train.transpose(1, 0)
   # y_train *= 0
    print(x_train.shape, y_train.shape)
    print(x_train[:,:,:,0])
    print(y_train)
    
    dims_full_input = [num_samples, 3, 32, 32]
    full_input = ffmodel.create_tensor_4d(dims_full_input, "", DataType.DT_FLOAT)

    dims_full_label = [num_samples, 1]
    full_label = ffmodel.create_tensor_2d(dims_full_label, "", DataType.DT_INT32)
    
    full_input.attach_numpy_array(ffconfig, x_train)
    full_label.attach_numpy_array(ffconfig, y_train)
    
    dataloader = DataLoader(ffmodel, alexnetconfig, input, label, full_input, full_label)
    dataloader.set_num_samples(10000)
  else:
    # Data Loader
    dataloader = DataLoader(ffmodel, alexnetconfig, input, label)

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

  ffoptimizer = SGDOptimizer(ffmodel, 0.0001)
  ffmodel.set_sgd_optimizer(ffoptimizer)

  ffmodel.init_layers()

 #  conv_2d1 = ffmodel.get_layer_by_id(11)
 #  cbias_tensor = conv_2d1.get_weight_tensor()
 #  input_tensor = conv_2d1.get_input_tensor_by_id(0)
 #  cbias_tensor.inline_map(ffconfig)
 #  cbias = cbias_tensor.get_array(ffconfig, DataType.DT_FLOAT)
 # # cbias += 0.125
 #  print(cbias.shape)
 #  #print(cbias)
 #  cbias_tensor.inline_unmap(ffconfig)



  epochs = ffconfig.get_epochs()

  ts_start = ffconfig.get_current_time()
  for epoch in range(0,epochs):
    dataloader.reset()
    ffmodel.reset_metrics()
    iterations = int(dataloader.get_num_samples() / ffconfig.get_batch_size())
    print(iterations, dataloader.get_num_samples())
    
    for iter in range(0, int(iterations)):
      # if (len(alexnetconfig.dataset_path) == 0):
      #   if (iter == 0 and epoch == 0):
      #     dataloader.next_batch(ffmodel)
      # else:
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
  #ffmodel.print_layers(13)

  conv_2d1 = ffmodel.get_layer_by_id(0)
  cbias_tensor = label#conv_2d1.get_input_tensor()
  #cbias_tensor = conv_2d1.get_output_tensor()
  cbias_tensor.inline_map(ffconfig)
  cbias = cbias_tensor.get_array(ffconfig, DataType.DT_INT32)
  print(cbias.shape)
  print(cbias)
  cbias_tensor.inline_unmap(ffconfig)
  
  if (use_external == True):
    full_input.detach_numpy_array(ffconfig)
    full_label.detach_numpy_array(ffconfig)

if __name__ == "__main__":
  print("alexnet")
  top_level_task()
