from flexflow.core import *

def top_level_task():
  ffconfig = FFConfig()
  ffconfig.parse_args()
  print("Python API batchSize(%d) workersPerNodes(%d) numNodes(%d)" %(ffconfig.get_batch_size(), ffconfig.get_workers_per_node(), ffconfig.get_num_nodes()))
  ffmodel = FFModel(ffconfig)
  
  dims_input = [ffconfig.get_batch_size(), 3, 229, 229]
  #print(dims)
  input = ffmodel.create_tensor_4d(dims_input, "", DataType.DT_FLOAT)
  
  dims_label = [ffconfig.get_batch_size(), 1]
  #print(dims)
  label = ffmodel.create_tensor_2d(dims_label, "", DataType.DT_INT32)
  
  conv1 = ffmodel.conv2d_v2("conv1", 3, 64, 11, 11, 4, 4, 2, 2)
  pool1 = ffmodel.pool2d_v2("pool1", 3, 3, 2, 2, 0, 0)
  conv2 = ffmodel.conv2d_v2("conv2", 64, 192, 5, 5, 1, 1, 2, 2)
  pool2 = ffmodel.pool2d_v2("pool2", 3, 3, 2, 2, 0, 0)
  conv3 = ffmodel.conv2d_v2("conv3", 192, 384, 3, 3, 1, 1, 1, 1)
  conv4 = ffmodel.conv2d_v2("conv4", 384, 256, 3, 3, 1, 1, 1, 1)
  conv5 = ffmodel.conv2d_v2("conv5", 256, 256, 3, 3, 1, 1, 1, 1)
  pool3 = ffmodel.pool2d_v2("pool3", 3, 3, 2, 2, 0, 0)
  flat = ffmodel.flat_v2("flat");
  linear1 = ffmodel.dense_v2("lienar1", 256*6*6, 4096, ActiMode.AC_MODE_RELU)
  linear2 = ffmodel.dense_v2("linear2", 4096, 4096, ActiMode.AC_MODE_RELU)
  linear3 = ffmodel.dense_v2("linear3", 4096, 1000)
  
  t = conv1.init_input(ffmodel, input);
  t = pool1.init_input(ffmodel, t);
  t = conv2.init_input(ffmodel, t);
  t = pool2.init_input(ffmodel, t);
  t = conv3.init_input(ffmodel, t);
  t = conv4.init_input(ffmodel, t);
  t = conv5.init_input(ffmodel, t);
  t = pool3.init_input(ffmodel, t);
  t = flat.init_input(ffmodel, t);
  t = linear1.init_input(ffmodel, t);
  t = linear2.init_input(ffmodel, t);
  t = linear3.init_input(ffmodel, t);
  t = ffmodel.softmax("softmax", t, label)
  
  ffoptimizer = SGDOptimizer(ffmodel, 0.01)
  ffmodel.set_sgd_optimizer(ffoptimizer)
  
  # Data Loader
  dataloader = DataLoader(ffmodel, input, label, 1)
  # input.inline_map(ffconfig)
  # input_array = input.get_array(ffconfig, DataType.DT_FLOAT)
  # input_array *= 1.0
  # print(input_array.shape)
  # input.inline_unmap(ffconfig)
  # label.inline_map(ffconfig)
  # label.inline_unmap(ffconfig)
  
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
    ffmodel.reset_metrics()
    iterations = 8192 / ffconfig.get_batch_size()
    for iter in range(0, int(iterations)):
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
  print("epochs %d, ELAPSED TIME = %.4fs, THROUGHPUT = %.2f samples/s\n" %(epochs, run_time, 8192 * epochs / run_time));
  #ffmodel.print_layers(13)
  
  conv_2d1 = ffmodel.get_layer_by_id(0)
  cbias_tensor = conv_2d1.get_weight_tensor()
  #cbias_tensor = conv_2d1.get_output_tensor()
  cbias_tensor.inline_map(ffconfig)
  cbias = cbias_tensor.get_array(ffconfig, DataType.DT_FLOAT)
  print(cbias.shape)
  #print(cbias)
  cbias_tensor.inline_unmap(ffconfig)

if __name__ == "__main__":
  print("alexnet")
  top_level_task()
