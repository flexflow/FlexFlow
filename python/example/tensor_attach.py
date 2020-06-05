from flexflow.core import *
import numpy as np

def top_level_task():
  ffconfig = FFConfig()
  alexnetconfig = NetConfig()
  print(alexnetconfig.dataset_path)
  ffconfig.parse_args()
  print("Python API batchSize(%d) workersPerNodes(%d) numNodes(%d)" %(ffconfig.get_batch_size(), ffconfig.get_workers_per_node(), ffconfig.get_num_nodes()))
  ffmodel = FFModel(ffconfig)
  
  dims_input = [8, 3, 10, 10]
  #print(dims)
  input = ffmodel.create_tensor(dims_input, "", DataType.DT_FLOAT)
  
  input_np = np.zeros((10,10,3,8), dtype=np.float32)
  ct = 0.0
  for i in range(0, input_np.shape[0]):
    for j in range(0, input_np.shape[1]):
      for k in range(0, input_np.shape[2]):
        for l in range(0, input_np.shape[3]):
          input_np[i, j, k, l] = ct
          ct += 1
  print(input_np)

  input.attach_numpy_array(ffconfig, input_np)
  print(input.is_mapped())
  input_array = input.get_array(ffconfig, DataType.DT_FLOAT)
  print(input_array)
  input.detach_numpy_array(ffconfig)


if __name__ == "__main__":
  print("tensor attach")
  top_level_task()
