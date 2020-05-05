from flexflow.keras.models import Model, init_internal_model, delete_internal_model
from flexflow.keras.layers import Flatten, Dense, Activation, Conv2D, MaxPooling2D
import flexflow.keras.optimizers
from flexflow.keras.datasets import mnist

import flexflow.core as ff
import numpy as np

import builtins
  
def mlp():
  num_classes = 10

  (x_train, y_train), (x_test, y_test) = mnist.load_data()

  x_train = x_train.reshape(60000, 784)
  x_train = x_train.astype('float32')
  x_train /= 255
  y_train = y_train.astype('int32')
  y_train = np.reshape(y_train, (len(y_train), 1))
  #y_train = np.random.randint(1, 9, size=(len(y_train),1), dtype='int32')
  print("shape: ", x_train.shape)
  
  dims_input = [builtins.internal_ffconfig.get_batch_size(), 784]
  input_tensor = builtins.internal_ffmodel.create_tensor_2d(dims_input, "", ff.DataType.DT_FLOAT);
  
  dims_label = [builtins.internal_ffconfig.get_batch_size(), 1]
  label_tensor = builtins.internal_ffmodel.create_tensor_2d(dims_label, "", ff.DataType.DT_INT32);
  
  output = Dense(512, input_shape=(784,), activation="relu")(input_tensor)
  output = Dense(512, activation="relu")(output)
  output = Dense(num_classes)(output)
  
  dense1 = builtins.internal_ffmodel.get_layer_by_id(0)
  dense2 = builtins.internal_ffmodel.get_layer_by_id(1)
  dense3 = builtins.internal_ffmodel.get_layer_by_id(2)
  
  model = Model(input_tensor, output)
  model.label_tensor = label_tensor
  
  model.add(dense1)
  model.add(dense2)
  model.add(dense3)
  
  output = model.add_softmax(output, label_tensor)

  opt = flexflow.keras.optimizers.SGD(learning_rate=0.01)
  model.compile(optimizer=opt)

  model.fit(x_train, y_train, epochs=1)
   
def cnn():
  num_classes = 10

  img_rows, img_cols = 28, 28
  
  (x_train, y_train), (x_test, y_test) = mnist.load_data()
  x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
  
  x_train = x_train.astype('float32')
  x_train /= 255
  y_train = y_train.astype('int32')
  y_train = np.reshape(y_train, (len(y_train), 1))
  
  dims_input = [builtins.internal_ffconfig.get_batch_size(), 1, 28, 28]
  input_tensor = builtins.internal_ffmodel.create_tensor_4d(dims_input, "", ff.DataType.DT_FLOAT);
  
  dims_label = [builtins.internal_ffconfig.get_batch_size(), 1]
  label_tensor = builtins.internal_ffmodel.create_tensor_2d(dims_label, "", ff.DataType.DT_INT32);
  
  output = Conv2D(filters=32, input_shape=(1,28,28), kernel_size=(3,3), strides=(1,1), padding=(1,1), activation="relu")(input_tensor)
  output = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding=(1,1), activation="relu")(output)
  output = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="valid")(output)
  output = Flatten()(output)
  output = Dense(128, activation="relu")(output)
  output = Dense(num_classes)(output)
  
  conv1 = builtins.internal_ffmodel.get_layer_by_id(0)
  conv2 = builtins.internal_ffmodel.get_layer_by_id(1)
  pool = builtins.internal_ffmodel.get_layer_by_id(2)
  flat = builtins.internal_ffmodel.get_layer_by_id(3)
  dense1 = builtins.internal_ffmodel.get_layer_by_id(4)
  dense2 = builtins.internal_ffmodel.get_layer_by_id(5)
  
  model = Model(input_tensor, output)
  model.label_tensor = label_tensor
  
  model.add(conv1)
  model.add(conv2)
  model.add(pool)
  model.add(flat)
  model.add(dense1)
  model.add(dense2)
  
  output = model.add_softmax(output, label_tensor)

  opt = flexflow.keras.optimizers.SGD(learning_rate=0.01)
  model.compile(optimizer=opt)

  model.fit(x_train, y_train, epochs=1)
  

def top_level_task():
  init_internal_model()
  
  cnn()
  #mlp()
  
  delete_internal_model()

if __name__ == "__main__":
  print("alexnet keras")
  top_level_task()