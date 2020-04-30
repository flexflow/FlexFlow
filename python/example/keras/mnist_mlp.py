from flexflow.keras.models import Sequential
from flexflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation
import flexflow.keras.optimizers
from flexflow.keras.datasets import mnist

import flexflow.core as ff

def top_level_task():
  
  num_classes = 10
  
  (x_train, y_train), (x_test, y_test) = mnist.load_data()
  
  x_train = x_train.reshape(60000, 784)
  x_train = x_train.astype('float32')
  x_train /= 255
  y_train = y_train.astype('int32')
  print("shape: ", x_train.shape)
  
  model = Sequential()
  model.add(Dense(512, input_shape=(784,), activation="relu"))
  model.add(Dense(512, activation="relu"))
  model.add(Dense(num_classes))
  model.add(Activation("softmax"))

  opt = flexflow.keras.optimizers.SGD(learning_rate=0.01)
  model.compile(optimizer=opt)

  model.fit(x_train, y_train, epochs=1)

if __name__ == "__main__":
  print("alexnet keras")
  top_level_task()