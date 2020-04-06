from flexflow.core import *

import numpy as np

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
  label = ffmodel.create_tensor_2d(dims_label, "", DataType.DT_INT32);
  
  t = ffmodel.conv2d("conv1", input, 64, 11, 11, 4, 4, 2, 2) 
  
  
  # Data Loader
  dataloader = DataLoader(ffmodel, input, label)
  
  ffmodel.init_layers()
  
  conv_2d1 = ffmodel.get_layer_by_id(0)
  conv_2d1.inline_map_weight(ffmodel)
  # raw_ptr = conv_2d1.get_weight_raw_ptr()
  # print(raw_ptr)
  # raw_ptr_int = int(ffi.cast("uintptr_t", raw_ptr))
  # shape = (11, 11, 3, 64)
  # strides = None
  # initializer = RegionNdarray(shape, "<f4", raw_ptr_int, strides, False)
  # array = np.asarray(initializer)
  weight = conv_2d1.get_weight()
  weight += 1.1
  print(weight)
  conv_2d1.inline_unmap_weight(ffmodel)
  
  conv_2d1.inline_map_bias(ffmodel)
  bias = conv_2d1.get_bias()
  bias += 1.2
  print(bias)
  conv_2d1.inline_unmap_bias(ffmodel)
  
  ffmodel.print_layers()

if __name__ == "__main__":
  print("alexnet")
  top_level_task()
