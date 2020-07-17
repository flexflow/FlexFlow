from flexflow.keras.models import Sequential, Model, Input
from flexflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, AveragePooling2D
import flexflow.keras.optimizers
from flexflow.keras.datasets import mnist

import flexflow.core as ff

import numpy as np

def top_level_task():
  
  num_classes = 10

  img_rows, img_cols = 28, 28
  
  (x_train, y_train), (x_test, y_test) = mnist.load_data()
  x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
  
  x_train = x_train.astype('float32')
  x_train /= 255
  y_train = y_train.astype('int32')
  y_train = np.reshape(y_train, (len(y_train), 1))
  print("shape: ", x_train.shape, x_train.__array_interface__["strides"])
  
  layers = [Conv2D(filters=32, input_shape=(1,28,28), kernel_size=(3,3), strides=(1,1), padding=(1,1), activation="relu"),
           Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding=(1,1), activation="relu"),
           MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="valid"),
           Flatten()]
  model1 = Sequential(layers)
  
  # layers2 = [Dense(128, input_shape=(12544,), activation="relu"),
  #          Dense(num_classes),
  #          Activation("softmax")]
  #
  # model2 = Sequential(layers2)
  #
  input_tensor = Input(shape=(12544,), dtype="float32")
  
  output = Dense(512, input_shape=(12544,), activation="relu")(input_tensor)
  output = Dense(num_classes)(output)
  output = Activation("softmax")(output)
  
  model2 = Model(input_tensor, output)
  
  model = Sequential()
  model.add(model1)
  model.add(model2)
  
  print(model.summary())

  opt = flexflow.keras.optimizers.SGD(learning_rate=0.01)
  model.compile(optimizer=opt)

  model.fit(x_train, y_train, epochs=1)

if __name__ == "__main__":
  print("alexnet keras")
  top_level_task()