from flexflow.keras.models import Sequential
from flexflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation
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
  
  # model = Sequential()
  # model.add(Conv2D(filters=32, input_shape=(1,28,28), kernel_size=(3,3), strides=(1,1), padding=(1,1), activation="relu"))
  # model.add(Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding=(1,1), activation="relu"))
  # model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="valid"))
  # model.add(Flatten())
  # model.add(Dense(128, activation="relu"))
  # model.add(Dense(num_classes))
  # model.add(Activation("softmax"))
  
  layers = [Conv2D(filters=32, input_shape=(1,28,28), kernel_size=(3,3), strides=(1,1), padding=(1,1), activation="relu"),
           Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding=(1,1), activation="relu"),
           MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="valid"),
           Flatten(),
           Dense(128, activation="relu"),
           Dense(num_classes),
           Activation("softmax")]
  model = Sequential(layers)
  
  print(model.summary())

  opt = flexflow.keras.optimizers.SGD(learning_rate=0.01)
  model.compile(optimizer=opt)

  model.fit(x_train, y_train, epochs=1)

if __name__ == "__main__":
  print("alexnet keras")
  top_level_task()