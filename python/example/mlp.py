from flexflow.core import *
import numpy as np

def top_level_task():
  ffconfig = FFConfig()
  ffconfig.parse_args()
  print("Python API batchSize(%d) workersPerNodes(%d) numNodes(%d)" %(ffconfig.get_batch_size(), ffconfig.get_workers_per_node(), ffconfig.get_num_nodes()))
  ffmodel = FFModel(ffconfig)
  
  dims1 = [ffconfig.get_batch_size(), 784]
  input1 = ffmodel.create_tensor_2d(dims1, "", DataType.DT_FLOAT);
  
  dims_label = [ffconfig.get_batch_size(), 1]
  label = ffmodel.create_tensor_2d(dims_label, "", DataType.DT_INT32);
 
  input1_np = np.empty((ffconfig.get_batch_size(), 784), dtype=np.float32)
  input1_np += 0.235
  input1.attach_numpy_array(ffconfig, input1_np)
  
  label_np = np.empty((ffconfig.get_batch_size(), 1), dtype=np.int32)
  label_np *= 0
  label.attach_numpy_array(ffconfig, label_np)
  
  t2 = ffmodel.dense("dense1", input1, 512, ActiMode.AC_MODE_RELU)
  t3 = ffmodel.dense("dense1", t2, 512, ActiMode.AC_MODE_RELU)
  t4 = ffmodel.dense("dense1", t3, 10, ActiMode.AC_MODE_RELU)
  t5 = ffmodel.softmax("softmax", t4, label)
  
  # Data Loader
 # dataloader = DataLoader(ffmodel, input1, label)
 
  
  ffoptimizer = SGDOptimizer(ffmodel, 0.01)
  ffmodel.set_sgd_optimizer(ffoptimizer)
  
  ffmodel.init_layers()
  
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
  
  dense1 = ffmodel.get_layer_by_id(0)

  dbias_tensor = dense1.get_bias_tensor()
  dbias_tensor.inline_map(ffconfig)
  dbias = dbias_tensor.get_array(ffconfig, DataType.DT_FLOAT)
  print(dbias.shape)
  #print(dbias)
  dbias_tensor.inline_unmap(ffconfig)

  dweight_tensor = dense1.get_output_tensor()
  dweight_tensor.inline_map(ffconfig)
  dweight = dweight_tensor.get_array(ffconfig, DataType.DT_FLOAT)
  print(dweight.shape)
  print(dweight)
  dweight_tensor.inline_unmap(ffconfig)
  
  input1.detach_numpy_array(ffconfig)
  label.detach_numpy_array(ffconfig)
  
if __name__ == "__main__":
  print("alexnet")
  top_level_task()
