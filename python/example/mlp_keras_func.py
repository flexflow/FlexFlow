from flexflow.keras.models import Model
from flexflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation

import flexflow.core as ff

import builtins

def top_level_task():
  print("hello")
  print("Python API batchSize(%d) workersPerNodes(%d) numNodes(%d)" %(builtins.internal_ffconfig.get_batch_size(), builtins.internal_ffconfig.get_workers_per_node(), builtins.internal_ffconfig.get_num_nodes()))
  
  dims = [builtins.internal_ffconfig.get_batch_size(), 784]
  input1 = builtins.internal_ffmodel.create_tensor_2d(dims, "", ff.DataType.DT_FLOAT);
  
  dims_label = [builtins.internal_ffconfig.get_batch_size(), 1]
  label = builtins.internal_ffmodel.create_tensor_2d(dims_label, "", ff.DataType.DT_INT32);
  
  input1.inline_map(builtins.internal_ffconfig)
  input1_array = input1.get_array(builtins.internal_ffconfig, ff.DataType.DT_FLOAT)
  print(input1_array.shape)
  input1.inline_unmap(builtins.internal_ffconfig)
  
  label.inline_map(builtins.internal_ffconfig)
  label_array = label.get_array(builtins.internal_ffconfig, ff.DataType.DT_INT32)
  print(label_array.shape)
  label.inline_unmap(builtins.internal_ffconfig)
  
  output = Dense(512, input_shape=(784,), activation="relu")(input1)
  output = Dense(512, activation="relu")(output)
  output = Dense(10, activation="relu")(output)
  
  model = Model(input1, output)
  
  del builtins.internal_ffconfig
  del builtins.internal_ffmodel

if __name__ == "__main__":
  print("alexnet keras")
  top_level_task()
