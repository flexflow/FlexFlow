from flexflow.core import *

def top_level_task():
  ffconfig = FFConfig()
  alexnetconfig = NetConfig()
  print(alexnetconfig.dataset_path)
  ffconfig.parse_args()
  print("Python API batchSize(%d) workersPerNodes(%d) numNodes(%d)" %(ffconfig.get_batch_size(), ffconfig.get_workers_per_node(), ffconfig.get_num_nodes()))
  ffmodel = FFModel(ffconfig)
  
  dims_input = [ffconfig.get_batch_size(), 3, 229, 229]
  #print(dims)
  input = ffmodel.create_tensor(dims_input, "", DataType.DT_FLOAT)
  
  dims_label = [ffconfig.get_batch_size(), 1]
  #print(dims)
  label = ffmodel.create_tensor(dims_label, "", DataType.DT_INT32)
  
  dataloader = DataLoader4D(ffmodel, input, label, ffnetconfig=alexnetconfig)
  
  conv1_1 = ffmodel.conv2d_v2("conv1", 3, 64, 11, 11, 4, 4, 2, 2)
  conv1_2 = ffmodel.conv2d_v2("conv1", 3, 64, 11, 11, 4, 4, 2, 2)
  pool1 = ffmodel.pool2d_v2("pool1", 3, 3, 2, 2, 0, 0)
  conv2 = ffmodel.conv2d_v2("conv2", 128, 192, 5, 5, 1, 1, 2, 2)
  pool2 = ffmodel.pool2d_v2("pool2", 3, 3, 2, 2, 0, 0)
  conv3 = ffmodel.conv2d_v2("conv3", 192, 384, 3, 3, 1, 1, 1, 1)
  conv4 = ffmodel.conv2d_v2("conv4", 384, 256, 3, 3, 1, 1, 1, 1)
  conv5 = ffmodel.conv2d_v2("conv5", 256, 256, 3, 3, 1, 1, 1, 1)
  pool3 = ffmodel.pool2d_v2("pool3", 3, 3, 2, 2, 0, 0)
  flat = ffmodel.flat_v2("flat");
  linear1 = ffmodel.dense_v2("lienar1", 256*6*6, 4096, ActiMode.AC_MODE_RELU)
  linear2 = ffmodel.dense_v2("linear2", 4096, 4096, ActiMode.AC_MODE_RELU)
  linear3 = ffmodel.dense_v2("linear3", 4096, 10)
  
  t1 = conv1_1.init_inout(ffmodel, input);
  t2 = conv1_2.init_inout(ffmodel, input);
  t = ffmodel.concat("concat", [t1, t2], 1)
  #t = conv1_1.init_inout(ffmodel, input);
  t = pool1.init_inout(ffmodel, t);
  t = conv2.init_inout(ffmodel, t);
  t = pool2.init_inout(ffmodel, t);
  t = conv3.init_inout(ffmodel, t);
  t = conv4.init_inout(ffmodel, t);
  t = conv5.init_inout(ffmodel, t);
  t = pool3.init_inout(ffmodel, t);
  t = flat.init_inout(ffmodel, t);
  t = linear1.init_inout(ffmodel, t);
  t = linear2.init_inout(ffmodel, t);
  t = linear3.init_inout(ffmodel, t);
  t = ffmodel.softmax("softmax", t, label)
  
  ffoptimizer = SGDOptimizer(ffmodel, 0.01)
  ffmodel.set_sgd_optimizer(ffoptimizer)
  
  # Data Loader

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
    dataloader.reset()
    ffmodel.reset_metrics()
    iterations = dataloader.get_num_samples() / ffconfig.get_batch_size()
    
    for iter in range(0, int(iterations)):
      if (len(alexnetconfig.dataset_path) == 0):
        if (iter == 0 and epoch == 0):
          dataloader.next_batch(ffmodel)
      else:
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
  
  conv_2d1 = ffmodel.get_layer_by_id(4)
  cbias_tensor = conv_2d1.get_weight_tensor()
  print(cbias_tensor)
  #cbias_tensor = conv_2d1.get_output_tensor()
  cbias_tensor.inline_map(ffconfig)
  cbias = cbias_tensor.get_array(ffconfig, DataType.DT_FLOAT)
  print(cbias.shape)
  #print(cbias)
  cbias_tensor.inline_unmap(ffconfig)
  
  label.inline_map(ffconfig)
  label_array = label.get_array(ffconfig, DataType.DT_INT32)
  print(label_array.shape)
  # print(cbias)
  print(label_array)
  label.inline_unmap(ffconfig)

if __name__ == "__main__":
  print("alexnet")
  top_level_task()
