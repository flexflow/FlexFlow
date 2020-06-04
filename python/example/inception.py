from flexflow.core import *

def InceptionA(ffmodel, input, pool_features, prefix=""):
  t1 = ffmodel.conv2d(prefix + "conv1", input, 64, 1, 1, 1, 1, 0, 0)
  t2 = ffmodel.conv2d(prefix + "ia_conv2", input, 48, 1, 1, 1, 1, 0, 0)
  t2 = ffmodel.conv2d(prefix + "conv3",t2, 64, 5, 5, 1, 1, 2, 2)
  t3 = ffmodel.conv2d(prefix + "conv4", input, 64, 1, 1, 1, 1, 0, 0)
  t3 = ffmodel.conv2d(prefix + "conv5", t3, 96, 3, 3, 1, 1, 1, 1)
  t3 = ffmodel.conv2d(prefix + "conv6", t3, 96, 3, 3, 1, 1, 1, 1)
  t4 = ffmodel.pool2d(prefix + "pool1", input, 3, 3, 1, 1, 1, 1, PoolType.POOL_AVG)
  t4 = ffmodel.conv2d(prefix + "conv7", t4, pool_features, 1, 1, 1, 1, 0, 0)
  output = ffmodel.concat(prefix + "concat1", [t1, t2, t3, t4], 1)
  return output

def InceptionB(ffmodel, input, prefix=""):
  t1 = ffmodel.conv2d(prefix + "conv1", input, 384, 3, 3, 2, 2, 0, 0)
  t2 = ffmodel.conv2d(prefix + "conv2", input, 64, 1, 1, 1, 1, 0, 0)
  t2 = ffmodel.conv2d(prefix + "conv3", t2, 96, 3, 3, 1, 1, 1, 1)
  t2 = ffmodel.conv2d(prefix + "conv4", t2, 96, 3, 3, 2, 2, 0, 0)
  t3 = ffmodel.pool2d(prefix + "pool1", input, 3, 3, 2, 2, 0, 0)
  output = ffmodel.concat(prefix + "concat1", [t1, t2, t3], 1)
  return output

def InceptionC(ffmodel, input, channels, prefix=""):
  t1 = ffmodel.conv2d(prefix + "conv1", input, 192, 1, 1, 1, 1, 0, 0)
  t2 = ffmodel.conv2d(prefix + "conv2", input, channels, 1, 1, 1, 1, 0, 0)
  t2 = ffmodel.conv2d(prefix + "conv3", t2, channels, 1, 7, 1, 1, 0, 3)
  t2 = ffmodel.conv2d(prefix + "conv4", t2, 192, 7, 1, 1, 1, 3, 0)
  t3 = ffmodel.conv2d(prefix + "pool1", input, channels, 1, 1, 1, 1, 0, 0)
  t3 = ffmodel.conv2d(prefix + "conv5", t3, channels, 7, 1, 1, 1, 3, 0)
  t3 = ffmodel.conv2d(prefix + "conv6", t3, channels, 1, 7, 1, 1, 0, 3)
  t3 = ffmodel.conv2d(prefix + "conv7", t3, channels, 7, 1, 1, 1, 3, 0)
  t3 = ffmodel.conv2d(prefix + "conv8", t3, 192, 1, 7, 1, 1, 0, 3)
  t4 = ffmodel.pool2d(prefix + "pool2", input, 3, 3, 1, 1, 1, 1, PoolType.POOL_AVG)
  t4 = ffmodel.conv2d(prefix + "conv9", t4, 192, 1, 1, 1, 1, 0, 0)
  output = ffmodel.concat(prefix + "concat1", [t1, t2, t3, t4], 1)
  return output;

def InceptionD(ffmodel, input, prefix=""):
  t1 = ffmodel.conv2d(prefix + "conv1", input, 192, 1, 1, 1, 1, 0, 0)
  t1 = ffmodel.conv2d(prefix + "conv2", t1, 320, 3, 3, 2, 2, 0, 0)
  t2 = ffmodel.conv2d(prefix + "conv3", input, 192, 1, 1, 1, 1, 0, 0)
  t2 = ffmodel.conv2d(prefix + "conv4", t2, 192, 1, 7, 1, 1, 0, 3)
  t2 = ffmodel.conv2d(prefix + "conv5", t2, 192, 7, 1, 1, 1, 3, 0)
  t2 = ffmodel.conv2d(prefix + "conv6", t2, 192, 3, 3, 2, 2, 0, 0)
  t3 = ffmodel.pool2d(prefix + "pool2", input, 3, 3, 2, 2, 0, 0)
  output = ffmodel.concat(prefix + "concat1", [t1, t2, t3], 1)
  return output;
  
def InceptionE(ffmodel, input, prefix=""):
  t1 = ffmodel.conv2d(prefix + "conv1", input, 320, 1, 1, 1, 1, 0, 0)
  t2i = ffmodel.conv2d(prefix + "conv2", input, 384, 1, 1, 1, 1, 0, 0)
  t2 = ffmodel.conv2d(prefix + "conv3", t2i, 384, 1, 3, 1, 1, 0, 1)
  t3 = ffmodel.conv2d(prefix + "conv4", t2i, 384, 3, 1, 1, 1, 1, 0)
  t3i = ffmodel.conv2d(prefix + "conv5", input, 448, 1, 1, 1, 1, 0, 0)
  t3i = ffmodel.conv2d(prefix + "conv6", t3i, 384, 3, 3, 1, 1, 1, 1)
  t4 = ffmodel.conv2d(prefix + "conv7", t3i, 384, 1, 3, 1, 1, 0, 1)
  t5 = ffmodel.conv2d(prefix + "conv8", t3i, 384, 3, 1, 1, 1, 1, 0)
  t6 = ffmodel.pool2d(prefix + "conv9", input, 3, 3, 1, 1, 1, 1, PoolType.POOL_AVG)
  t6 = ffmodel.conv2d(prefix + "conv10", t6, 192, 1, 1, 1, 1, 0, 0)
  output = ffmodel.concat(prefix + "concat1", [t1, t2, t3, t4, t5, t6], 1)
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
  
  t = ffmodel.conv2d("conv1", input, 32, 3, 3, 2, 2, 0, 0)
  t = ffmodel.conv2d("conv2", t, 32, 3, 3, 1, 1, 0, 0)
  t = ffmodel.conv2d("conv3", t, 64, 3, 3, 1, 1, 1, 1)
  t = ffmodel.pool2d("pool1", t, 3, 3, 2, 2, 0, 0)
  t = ffmodel.conv2d("conv4", t, 80, 1, 1, 1, 1, 0, 0)
  t = ffmodel.conv2d("conv5", t, 192, 3, 3, 1, 1, 1, 1)
  t = ffmodel.pool2d("pool1", t, 3, 3, 2, 2, 0, 0)
  t = InceptionA(ffmodel, t, 32, "ia1_")
  t = InceptionA(ffmodel, t, 64, "ia2_")
  t = InceptionA(ffmodel, t, 64, "ia3_")
  t = InceptionB(ffmodel, t, "ib1_")
  t = InceptionC(ffmodel, t, 128, "ic1_")
  t = InceptionC(ffmodel, t, 160, "ic2_")
  t = InceptionC(ffmodel, t, 160, "ic3_")
  t = InceptionC(ffmodel, t, 192, "ic4_")
  t = InceptionD(ffmodel, t, "id1_")
  t = InceptionE(ffmodel, t, "ie1_")
  t = InceptionE(ffmodel, t, "ie1_")
  t = ffmodel.pool2d("pool1", t, 8, 8, 1, 1, 0, 0, PoolType.POOL_AVG)
  t = ffmodel.flat("flat", t)
  t = ffmodel.dense("linear1",t, 1000)
  t = ffmodel.softmax("softmax", t, label)
  
  ffoptimizer = SGDOptimizer(ffmodel, 0.01)
  ffmodel.set_sgd_optimizer(ffoptimizer)
  
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
