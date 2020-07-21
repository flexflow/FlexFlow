from flexflow.core import *

import numpy as np

def top_level_task():
  ffconfig = FFConfig()
  ffconfig.parse_args()
  print("Python API batchSize(%d) workersPerNodes(%d) numNodes(%d)" %(ffconfig.get_batch_size(), ffconfig.get_workers_per_node(), ffconfig.get_num_nodes()))
  ffmodel = FFModel(ffconfig)
  
  dims1 = [ffconfig.get_batch_size(), 3, 229, 229]
  input1 = ffmodel.create_tensor(dims1, "", DataType.DT_FLOAT);
  
  dims2 = [ffconfig.get_batch_size(), 256]
  input2 = ffmodel.create_tensor(dims2, "", DataType.DT_FLOAT);
  
  dims_label = [ffconfig.get_batch_size(), 1]
  label = ffmodel.create_tensor(dims_label, "", DataType.DT_INT32);
  
  alexnetconfig = NetConfig()
  dataloader = DataLoader4D(ffmodel, input1, label, ffnetconfig=alexnetconfig)
  dataloader.reset()
  dataloader.next_batch(ffmodel)

  input1.inline_map(ffconfig)
  input1.dims = [64, 3, 229, 229]
  input1_array = input1.get_array(ffconfig, DataType.DT_FLOAT)
  # input1_array *= 0
  # input1_array += 1.1
  print(input1_array.shape)
  print(input1_array)
  input1.inline_unmap(ffconfig)
  
  input2.inline_map(ffconfig)
  input2_array = input2.get_array(ffconfig, DataType.DT_FLOAT)
  input2_array *= 0
  input2_array += 2.2
  print(input2_array.shape)
  print(input2_array)
  input2.inline_unmap(ffconfig)
  
  input1 = ffmodel.conv2d(input1, 64, 11, 11, 4, 4, 2, 2) 
  input2 = ffmodel.dense(input2, 128, ActiMode.AC_MODE_RELU)
  input2 = ffmodel.dense(input2, 128, ActiMode.AC_MODE_RELU)

  
  ffmodel.init_layers()
  
  conv1 = ffmodel.get_layer_by_id(0)
  input_tensor1 = conv1.get_input_tensor()
  input_tensor1.inline_map(ffconfig)
  input_array11 = input_tensor1.get_array(ffconfig, DataType.DT_FLOAT)
  print(input_array11.shape)
  #print(input_array11)
  input_tensor1.inline_unmap(ffconfig)
  
  output_tensor1 = conv1.get_output_tensor()
  output_tensor1.inline_map(ffconfig)
  output_array11 = output_tensor1.get_array(ffconfig, DataType.DT_FLOAT)
  print(output_array11.shape)
  #print(output_array11)
  output_tensor1.inline_unmap(ffconfig)
  
  
  dense1 = ffmodel.get_layer_by_id(1)
  input_tensor2 = dense1.get_input_tensor()
  input_tensor2.inline_map(ffconfig)
  input_array22 = input_tensor2.get_array(ffconfig, DataType.DT_FLOAT)
  print(input_array22.shape)
  print(input_array22)
  input_tensor2.inline_unmap(ffconfig)
  
  output_tensor2 = dense1.get_output_tensor()
  output_tensor2.inline_map(ffconfig)
  output_array22 = output_tensor2.get_array(ffconfig, DataType.DT_FLOAT)
  print(output_array22.shape)
  #print(output_array11)
  output_tensor2.inline_unmap(ffconfig)


if __name__ == "__main__":
  print("alexnet")
  top_level_task()
