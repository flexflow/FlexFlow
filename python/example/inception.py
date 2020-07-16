from flexflow.core import *

def InceptionA(ffmodel, input, pool_features):
  t1 = ffmodel.conv2d(input, 64, 1, 1, 1, 1, 0, 0)
  t2 = ffmodel.conv2d(input, 48, 1, 1, 1, 1, 0, 0)
  t2 = ffmodel.conv2d(t2, 64, 5, 5, 1, 1, 2, 2)
  t3 = ffmodel.conv2d(input, 64, 1, 1, 1, 1, 0, 0)
  t3 = ffmodel.conv2d(t3, 96, 3, 3, 1, 1, 1, 1)
  t3 = ffmodel.conv2d(t3, 96, 3, 3, 1, 1, 1, 1)
  t4 = ffmodel.pool2d(input, 3, 3, 1, 1, 1, 1, PoolType.POOL_AVG)
  t4 = ffmodel.conv2d(t4, pool_features, 1, 1, 1, 1, 0, 0)
  output = ffmodel.concat([t1, t2, t3, t4], 1)
  return output

def InceptionB(ffmodel, input):
  t1 = ffmodel.conv2d(input, 384, 3, 3, 2, 2, 0, 0)
  t2 = ffmodel.conv2d(input, 64, 1, 1, 1, 1, 0, 0)
  t2 = ffmodel.conv2d(t2, 96, 3, 3, 1, 1, 1, 1)
  t2 = ffmodel.conv2d(t2, 96, 3, 3, 2, 2, 0, 0)
  t3 = ffmodel.pool2d(input, 3, 3, 2, 2, 0, 0)
  output = ffmodel.concat([t1, t2, t3], 1)
  return output

def InceptionC(ffmodel, input, channels):
  t1 = ffmodel.conv2d(input, 192, 1, 1, 1, 1, 0, 0)
  t2 = ffmodel.conv2d(input, channels, 1, 1, 1, 1, 0, 0)
  t2 = ffmodel.conv2d(t2, channels, 1, 7, 1, 1, 0, 3)
  t2 = ffmodel.conv2d(t2, 192, 7, 1, 1, 1, 3, 0)
  t3 = ffmodel.conv2d(input, channels, 1, 1, 1, 1, 0, 0)
  t3 = ffmodel.conv2d(t3, channels, 7, 1, 1, 1, 3, 0)
  t3 = ffmodel.conv2d(t3, channels, 1, 7, 1, 1, 0, 3)
  t3 = ffmodel.conv2d(t3, channels, 7, 1, 1, 1, 3, 0)
  t3 = ffmodel.conv2d(t3, 192, 1, 7, 1, 1, 0, 3)
  t4 = ffmodel.pool2d(input, 3, 3, 1, 1, 1, 1, PoolType.POOL_AVG)
  t4 = ffmodel.conv2d(t4, 192, 1, 1, 1, 1, 0, 0)
  output = ffmodel.concat([t1, t2, t3, t4], 1)
  return output;

def InceptionD(ffmodel, input):
  t1 = ffmodel.conv2d(input, 192, 1, 1, 1, 1, 0, 0)
  t1 = ffmodel.conv2d(t1, 320, 3, 3, 2, 2, 0, 0)
  t2 = ffmodel.conv2d(input, 192, 1, 1, 1, 1, 0, 0)
  t2 = ffmodel.conv2d(t2, 192, 1, 7, 1, 1, 0, 3)
  t2 = ffmodel.conv2d(t2, 192, 7, 1, 1, 1, 3, 0)
  t2 = ffmodel.conv2d(t2, 192, 3, 3, 2, 2, 0, 0)
  t3 = ffmodel.pool2d(input, 3, 3, 2, 2, 0, 0)
  output = ffmodel.concat([t1, t2, t3], 1)
  return output;
  
def InceptionE(ffmodel, input):
  t1 = ffmodel.conv2d(input, 320, 1, 1, 1, 1, 0, 0)
  t2i = ffmodel.conv2d(input, 384, 1, 1, 1, 1, 0, 0)
  t2 = ffmodel.conv2d(t2i, 384, 1, 3, 1, 1, 0, 1)
  t3 = ffmodel.conv2d(t2i, 384, 3, 1, 1, 1, 1, 0)
  t3i = ffmodel.conv2d(input, 448, 1, 1, 1, 1, 0, 0)
  t3i = ffmodel.conv2d(t3i, 384, 3, 3, 1, 1, 1, 1)
  t4 = ffmodel.conv2d(t3i, 384, 1, 3, 1, 1, 0, 1)
  t5 = ffmodel.conv2d(t3i, 384, 3, 1, 1, 1, 1, 0)
  t6 = ffmodel.pool2d(input, 3, 3, 1, 1, 1, 1, PoolType.POOL_AVG)
  t6 = ffmodel.conv2d(t6, 192, 1, 1, 1, 1, 0, 0)
  output = ffmodel.concat([t1, t2, t3, t4, t5, t6], 1)
  return output;
  
def top_level_task():
  ffconfig = FFConfig()
  ffconfig.parse_args()
  print("Python API batchSize(%d) workersPerNodes(%d) numNodes(%d)" %(ffconfig.get_batch_size(), ffconfig.get_workers_per_node(), ffconfig.get_num_nodes()))
  ffmodel = FFModel(ffconfig)
  
  dims_input = [ffconfig.get_batch_size(), 3, 299, 299]
  #print(dims)
  input = ffmodel.create_tensor(dims_input, "", DataType.DT_FLOAT)
  
  dims_label = [ffconfig.get_batch_size(), 1]
  #print(dims)
  label = ffmodel.create_tensor(dims_label, "", DataType.DT_INT32)
  
  t = ffmodel.conv2d(input, 32, 3, 3, 2, 2, 0, 0)
  t = ffmodel.conv2d(t, 32, 3, 3, 1, 1, 0, 0)
  t = ffmodel.conv2d(t, 64, 3, 3, 1, 1, 1, 1)
  t = ffmodel.pool2d(t, 3, 3, 2, 2, 0, 0)
  t = ffmodel.conv2d(t, 80, 1, 1, 1, 1, 0, 0)
  t = ffmodel.conv2d(t, 192, 3, 3, 1, 1, 1, 1)
  t = ffmodel.pool2d(t, 3, 3, 2, 2, 0, 0)
  t = InceptionA(ffmodel, t, 32)
  t = InceptionA(ffmodel, t, 64)
  t = InceptionA(ffmodel, t, 64)
  t = InceptionB(ffmodel, t)
  t = InceptionC(ffmodel, t, 128)
  t = InceptionC(ffmodel, t, 160)
  t = InceptionC(ffmodel, t, 160)
  t = InceptionC(ffmodel, t, 192)
  t = InceptionD(ffmodel, t)
  t = InceptionE(ffmodel, t)
  t = InceptionE(ffmodel, t)
  t = ffmodel.pool2d(t, 8, 8, 1, 1, 0, 0, PoolType.POOL_AVG)
  t = ffmodel.flat(t)
  t = ffmodel.dense(t, 1000)
  t = ffmodel.softmax(t, label)
  
  ffoptimizer = SGDOptimizer(ffmodel, 0.01)
  ffmodel.set_sgd_optimizer(ffoptimizer)
  ffmodel.compile()
  
  # Data Loader
  alexnetconfig = NetConfig()
  dataloader = DataLoader4D(ffmodel, input, label, ffnetconfig=alexnetconfig)
  dataloader.set_num_samples(256 * ffconfig.get_workers_per_node() * ffconfig.get_num_nodes())
  
  ffmodel.init_layers()
  
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
        ffconfig.start_trace(111)
      ffmodel.forward()
      ffmodel.zero_gradients()
      ffmodel.backward()
      ffmodel.update()
      if (epoch > 0):
        ffconfig.end_trace(111)

  ts_end = ffconfig.get_current_time()
  run_time = 1e-6 * (ts_end - ts_start);
  print("epochs %d, ELAPSED TIME = %.4fs, THROUGHPUT = %.2f samples/s\n" %(epochs, run_time, 8192 * epochs / run_time));
  
  # conv_2d1 = ffmodel.get_layer_by_id(7)
  # cbias_tensor = conv_2d1.get_weight_tensor()
  # print(cbias_tensor)
  # #cbias_tensor = conv_2d1.get_output_tensor()
  # cbias_tensor.inline_map(ffconfig)
  # cbias = cbias_tensor.get_array(ffconfig, DataType.DT_FLOAT)
  # print(cbias.shape)
  # #print(cbias)
  # cbias_tensor.inline_unmap(ffconfig)

if __name__ == "__main__":
  print("alexnet")
  top_level_task()
