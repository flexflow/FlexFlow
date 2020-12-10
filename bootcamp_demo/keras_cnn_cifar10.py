#./flexflow_python $FF_HOME/bootcamp_demo/keras_cnn_cifar10.py -ll:py 1 -ll:gpu 1 -ll:fsize 2048 -ll:zsize 12192

# from keras.models import Model, Sequential
# from keras.layers import Input, Flatten, Dense, Activation, Conv2D, MaxPooling2D, Dropout
# from keras.optimizers import SGD
# from keras.datasets import cifar10
# from keras import losses
# from keras import metrics

from flexflow.keras.models import Model, Sequential
from flexflow.keras.layers import Input, Flatten, Dense, Activation, Conv2D, MaxPooling2D, Dropout
from flexflow.keras.optimizers import SGD
from flexflow.keras.datasets import cifar10
from flexflow.keras import losses
from flexflow.keras import metrics

import numpy as np
  
def top_level_task():
  num_classes = 10
  
  num_samples = 10000
  
  #(x_train, y_train), (x_test, y_test) = cifar10.load_data()
  (x_train, y_train), (x_test, y_test) = cifar10.load_data(num_samples)
  
  x_train = x_train.astype('float32')
  x_train /= 255
  y_train = y_train.astype('int32')
  print("shape: ", x_train.shape[1:])
  
  model = Sequential()

  model.add(Conv2D(filters=32, input_shape=(3,32,32), kernel_size=(3,3), strides=(1,1), padding="valid", activation="relu"))
  model.add(Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding="valid", activation="relu"))
  model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="valid"))
  model.add(Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding="valid", activation="relu"))
  model.add(Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding="valid"))
  model.add(Activation("relu"))
  model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="valid"))
  model.add(Flatten())
  model.add(Dense(512))
  model.add(Activation("relu"))
  model.add(Dropout(0.5))
  model.add(Dense(num_classes))
  model.add(Activation("softmax"))
  
  opt = SGD(learning_rate=0.01)
  model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy', 'sparse_categorical_crossentropy'])
  print(model.summary())

  model.fit(x_train, y_train, batch_size=64, epochs=4)

if __name__ == "__main__":
  print("Functional API, cifar10 cnn")
  top_level_task()