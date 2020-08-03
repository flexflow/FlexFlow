from flexflow.keras.models import Model, Sequential
from flexflow.keras.layers import Input, Flatten, Dense, Activation, Conv2D, MaxPooling2D, Concatenate, concatenate
import flexflow.keras.optimizers
from flexflow.keras.datasets import mnist
from flexflow.keras.datasets import cifar10
from flexflow.keras import losses
from flexflow.keras import metrics

import flexflow.core as ff
import numpy as np
import argparse
import gc
  
def top_level_task():
  num_classes = 10
  
  num_samples = 10000
  
  (x_train, y_train), (x_test, y_test) = cifar10.load_data(num_samples)
  
  x_train = x_train.astype('float32')
  x_train /= 255
  y_train = y_train.astype('int32')
  print("shape: ", x_train.shape)
  
  input_tensor1 = Input(shape=(3, 32, 32), dtype="float32")
  
  output_tensor = Conv2D(filters=32, input_shape=(3,32,32), kernel_size=(3,3), strides=(1,1), padding=(1,1), activation="relu")(input_tensor1)
  output_tensor = Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding=(1,1), activation="relu")(output_tensor)
  output_tensor = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="valid")(output_tensor)
  output_tensor = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding=(1,1), activation="relu")(output_tensor)
  output_tensor = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding=(1,1), activation="relu")(output_tensor)
  output_tensor = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="valid")(output_tensor)
  output_tensor = Flatten()(output_tensor)
  output_tensor = Dense(512, activation="relu")(output_tensor)
  output_tensor = Dense(num_classes)(output_tensor)
  output_tensor = Activation("softmax")(output_tensor)

  model = Model(input_tensor1, output_tensor)
  
  opt = flexflow.keras.optimizers.SGD(learning_rate=0.01)
  model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy', 'sparse_categorical_crossentropy'])
  print(model.summary())

  model.fit(x_train, y_train, epochs=1)

if __name__ == "__main__":
  print("Functional API, cifar10 cnn")
  top_level_task()
  gc.collect()