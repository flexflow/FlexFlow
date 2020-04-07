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
  
  # input.inline_map(ffconfig)
  # input_array = input.get_array_float(ffconfig)
  # input_array += 1.2
  # print(input_array.shape)
  # input.inline_unmap(ffconfig)
  
  t1 = ffmodel.get_tensor_by_id(1)
  t1.inline_map(ffconfig)
  t1_array = t1.get_array_float(ffconfig)
  t1_array += 1.2
  print(t1_array)
  t1.inline_unmap(ffconfig)
  
  conv_2d1 = ffmodel.get_layer_by_id(0)

  bias_tensor = conv_2d1.get_bias_tensor()
  bias_tensor.inline_map(ffconfig)
  bias = bias_tensor.get_array_float(ffconfig)
  bias += 1.2
  print(bias)
  bias_tensor.inline_unmap(ffconfig)

  weight_tensor = conv_2d1.get_weight_tensor()
  weight_tensor.inline_map(ffconfig)
  weight = weight_tensor.get_array_float(ffconfig)
  weight += 1.1
  print(weight)
  weight_tensor.inline_unmap(ffconfig)
  
  weight_tensor.inline_map(ffconfig)
  weight2 = weight_tensor.get_array_float(ffconfig)
  weight2 += 1.1
  #print(weight)
  weight_tensor.inline_unmap(ffconfig)
  
  #ffmodel.print_layers()

if __name__ == "__main__":
  print("alexnet")
  top_level_task()
