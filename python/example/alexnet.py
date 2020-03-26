from flexflow.core import *

def top_level_task():
  ffconfig = FFConfig()
  ffconfig.parse_args()
  print("Python API batchSize(%d) workersPerNodes(%d) numNodes(%d)" %(ffconfig.get_batch_size(), ffconfig.get_workers_per_node(), ffconfig.get_num_nodes()))
  ffmodel = FFModel(ffconfig)
  
  dims_input = [ffconfig.get_batch_size(), 3, 229, 229]
  #print(dims)
  input = ffmodel.create_tensor_4d(dims_input, "", DataType.DT_FLOAT);
  
  dims_label = [ffconfig.get_batch_size(), 1]
  #print(dims)
  label = ffmodel.create_tensor_2d(dims_label, "", DataType.DT_INT32);
  
  # ts0 = ffmodel.conv2d("conv1", input, 64, 11, 11, 4, 4, 2, 2)
  # ts1 = ffmodel.conv2d("conv1", input, 64, 11, 11, 4, 4, 2, 2)
  # t1 = ffmodel.concat("concat", [ts0, ts1], 1)
  # t2 = ffmodel.pool2d("pool1", t1, 3, 3, 2, 2, 0, 0)
  # t3 = ffmodel.conv2d("conv2", t2, 192, 5, 5, 1, 1, 2, 2)
  # t4 = ffmodel.pool2d("pool2", t3, 3, 3, 2, 2, 0, 0)
  # t5 = ffmodel.conv2d("conv3", t4, 384, 3, 3, 1, 1, 1, 1)
  # t6 = ffmodel.conv2d("conv4", t5, 256, 3, 3, 1, 1, 1, 1)
  # t7 = ffmodel.conv2d("conv5", t6, 256, 3, 3, 1, 1, 1, 1)
  # t8 = ffmodel.pool2d("pool3", t7, 3, 3, 2, 2, 0, 0)
  # t9 = ffmodel.flat("flat", t8);
  # t10 = ffmodel.dense("lienar1", t9, 4096, ActiMode.AC_MODE_RELU);
  # t11 = ffmodel.dense("linear2", t10, 4096, ActiMode.AC_MODE_RELU);
  # t12 = ffmodel.dense("linear3", t11, 1000)
  # t13 = ffmodel.softmax("softmax", t12, label)
  ts0 = ffmodel.conv2d("conv1", input, 64, 11, 11, 4, 4, 2, 2)
  ts1 = ffmodel.conv2d("conv1", input, 64, 11, 11, 4, 4, 2, 2)
  t = ffmodel.concat("concat", [ts0, ts1], 1)
  t = ffmodel.pool2d("pool1", t, 3, 3, 2, 2, 0, 0)
  t = ffmodel.conv2d("conv2", t, 192, 5, 5, 1, 1, 2, 2)
  t = ffmodel.pool2d("pool2", t, 3, 3, 2, 2, 0, 0)
  t = ffmodel.conv2d("conv3", t, 384, 3, 3, 1, 1, 1, 1)
  t = ffmodel.conv2d("conv4", t, 256, 3, 3, 1, 1, 1, 1)
  t = ffmodel.conv2d("conv5", t, 256, 3, 3, 1, 1, 1, 1)
  t = ffmodel.pool2d("pool3", t, 3, 3, 2, 2, 0, 0)
  t = ffmodel.flat("flat", t);
  t = ffmodel.dense("lienar1", t, 4096, ActiMode.AC_MODE_RELU);
  t = ffmodel.dense("linear2", t, 4096, ActiMode.AC_MODE_RELU);
  t = ffmodel.dense("linear3", t, 1000)
  t = ffmodel.softmax("softmax", t, label)
  
  ffoptimizer = SGDOptimizer(ffmodel, 0.01)
  ffmodel.set_sgd_optimizer(ffoptimizer)
  
  # Data Loader
  dataloader = DataLoader(ffmodel, input, label)
  
  ffmodel.init_layers()
  
  epochs = ffconfig.get_epochs()
  
  ts_start = ffconfig.get_current_time()
  for epoch in range(0,epochs):
    ffmodel.reset_metrics()
    iterations = 8192 / ffconfig.get_batch_size()
    for iter in range(0, int(iterations)):
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

if __name__ == "__main__":
  print("alexnet")
  top_level_task()
