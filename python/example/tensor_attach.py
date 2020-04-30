from flexflow.core import *
import numpy as np

def top_level_task():
  ffconfig = FFConfig()
  alexnetconfig = NetConfig()
  print(alexnetconfig.dataset_path)
  ffconfig.parse_args()
  print("Python API batchSize(%d) workersPerNodes(%d) numNodes(%d)" %(ffconfig.get_batch_size(), ffconfig.get_workers_per_node(), ffconfig.get_num_nodes()))
  ffmodel = FFModel(ffconfig)
  
  dims_input = [ffconfig.get_batch_size(), 3, 229, 229]
  #print(dims)
  input = ffmodel.create_tensor_4d(dims_input, "", DataType.DT_FLOAT)

  dims_label = [ffconfig.get_batch_size(), 32]
  #print(dims)
  label = ffmodel.create_tensor_2d(dims_label, "", DataType.DT_INT32)
  
  label_np = np.empty((ffconfig.get_batch_size(), 32), dtype='int32')
  label_np += 1
  
  label.attach_numpy_array(ffconfig, label_np)
  print(label.is_mapped())
  label_array = label.get_array(ffconfig, DataType.DT_INT32)
  print(label_array)
  label.detach_numpy_array(ffconfig)


if __name__ == "__main__":
  print("tensor attach")
  top_level_task()
