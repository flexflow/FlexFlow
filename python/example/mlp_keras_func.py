from flexflow.keras.models import Model, init_internal_model, delete_internal_model
from flexflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation

import flexflow.core as ff

import builtins

def top_level_task():
  init_internal_model()
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
  
  # output = Dense(512, input_shape=(784,), activation="relu")(input1)
  # output = Dense(512, activation="relu")(output)
  # output = Dense(10, activation="relu")(output)
  #
  # dense1 = builtins.internal_ffmodel.get_layer_by_id(0)
  # dense2 = builtins.internal_ffmodel.get_layer_by_id(1)
  # dense3 = builtins.internal_ffmodel.get_layer_by_id(2)
  #
  # model = Model(input1, output)
  # model.add(dense1)
  # model.add(dense2)
  # model.add(dense3)
  
  dims2 = [builtins.internal_ffconfig.get_batch_size(), 3, 229, 229]
  input2 = builtins.internal_ffmodel.create_tensor_4d(dims2, "", ff.DataType.DT_FLOAT);
  alexnetconfig = ff.NetConfig()
  dataloader = ff.DataLoader(builtins.internal_ffmodel, alexnetconfig, input2, label)
  
  output = Conv2D(filters=64, input_shape=(229,229,3), kernel_size=(11,11), strides=(4,4), padding=(2,2))(input2)
  output = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding="valid")(output)
  output = Conv2D(filters=192, kernel_size=(5,5), strides=(1,1), padding=(2,2))(output)
  output = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding="valid")(output)
  output = Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding=(1,1))(output)
  output = Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding=(1,1))(output)
  output = Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding=(1,1))(output)
  output = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding="valid")(output)
  output = Flatten()(output)
  output = Dense(4096, activation="relu")(output)
  output = Dense(4096, activation="relu")(output)
  output = Dense(1000)(output)
  
  conv1 = builtins.internal_ffmodel.get_layer_by_id(0)
  pool1 = builtins.internal_ffmodel.get_layer_by_id(1)
  conv2 = builtins.internal_ffmodel.get_layer_by_id(2)
  pool2 = builtins.internal_ffmodel.get_layer_by_id(3)
  conv3 = builtins.internal_ffmodel.get_layer_by_id(4)
  conv4 = builtins.internal_ffmodel.get_layer_by_id(5)
  conv5 = builtins.internal_ffmodel.get_layer_by_id(6)
  pool3 = builtins.internal_ffmodel.get_layer_by_id(7)
  flat = builtins.internal_ffmodel.get_layer_by_id(8)
  dense1 = builtins.internal_ffmodel.get_layer_by_id(9)
  dense2 = builtins.internal_ffmodel.get_layer_by_id(10)
  dense3 = builtins.internal_ffmodel.get_layer_by_id(11)
  
  model = Model(input2, output)
  
  model.add(conv1)
  model.add(pool1)
  model.add(conv2)
  model.add(pool2)
  model.add(conv3)
  model.add(conv4)
  model.add(conv5)
  model.add(pool3)
  model.add(flat)
  model.add(dense1)
  model.add(dense2)
  model.add(dense3)
  
  output = model.add_softmax(output, label)

  model.compile()
  model.fit(input2, label)
  
  delete_internal_model()

if __name__ == "__main__":
  print("alexnet keras")
  top_level_task()
