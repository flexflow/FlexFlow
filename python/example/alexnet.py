from flexflow.core import *

def top_level_task():
  ffconfig = FFConfig()
  ffconfig.parse_args()
  print("Python API batchSize(%d) workersPerNodes(%d) numNodes(%d)" %(ffconfig.get_batch_size(), ffconfig.get_workers_per_node(), ffconfig.get_num_nodes()))
  ffmodel = FFModel(ffconfig)
  
  dims = [ffconfig.get_batch_size(), 3, 229, 229]
  #print(dims)
  input = ffmodel.create_tensor_4d(dims, "", DataType.DT_FLOAT);
  
  dims_label = [ffconfig.get_batch_size(), 1]
  #print(dims)
  label = ffmodel.create_tensor_2d(dims_label, "", DataType.DT_FLOAT);
  
  t1 = ffmodel.conv2d("conv1", input, 64, 11, 11, 4, 4, 2, 2)
  t2 = ffmodel.pool2d("pool1", t1, 3, 3, 2, 2, 0, 0)
  t3 = ffmodel.conv2d("conv2", t2, 192, 5, 5, 1, 1, 2, 2)
  t4 = ffmodel.pool2d("pool2", t3, 3, 3, 2, 2, 0, 0)
  t5 = ffmodel.conv2d("conv3", t4, 384, 3, 3, 1, 1, 1, 1)
  t6 = ffmodel.conv2d("conv4", t5, 256, 3, 3, 1, 1, 1, 1)
  t7 = ffmodel.conv2d("conv5", t6, 256, 3, 3, 1, 1, 1, 1)
  t8 = ffmodel.pool2d("pool3", t7, 3, 3, 2, 2, 0, 0)
  t9 = ffmodel.flat("flat", t8);
  t10 = ffmodel.linear("lienar1", t9, 4096, ActiMode.AC_MODE_RELU);
  t11 = ffmodel.linear("linear2", t10, 4096, ActiMode.AC_MODE_RELU);
  t12 = ffmodel.linear("linear3", t11, 1000)
  #t13 = ffmodel.softmax("softmax", t12)
  
  ffoptimizer = SGDOptimizer(ffmodel, 0.01)
  ffmodel.set_sgd_optimizer(ffoptimizer)
  
  # Data Loader
  dataloader = DataLoader(ffmodel, input, label)
  
  ffmodel.init_layers()
  
  for epoch in range(0,1):
    ffmodel.reset_metrics()
    iterations = 8192 / ffconfig.get_batch_size()
    print(iterations)
  #  for iter in range(0, int(iterations)):
   #   ffmodel.forward()
      #ffmodel.zero_gradidents()
      #ffmodel.backward()
      #ffmodel.update

if __name__ == "__main__":
  print("alexnet")
  top_level_task()
