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

  (x_train, y_train), (x_test, y_test) = mnist.load_data()

  x_train = x_train.reshape(60000, 784)
  x_train = x_train.astype('float32')
  x_train /= 255
  y_train = y_train.astype('int32')
  y_train = np.reshape(y_train, (len(y_train), 1))
  print("shape: ", x_train.shape)
  
  input_tensor1 = Input(shape=(256,))
  input_tensor2 = Input(shape=(256,))
  input_tensor3 = Input(shape=(256,))
  input_tensor4 = Input(shape=(256,))
  
  t1 = Dense(512, activation="relu", name="dense1")(input_tensor1)
  t1 = Dense(512, activation="relu", name="dense12")(t1)
  model1 = Model(input_tensor1, t1)
  t2 = Dense(512, activation="relu", name="dense2")(input_tensor2)
  t2 = Dense(512, activation="relu", name="dense22")(t2)
  model2 = Model(input_tensor2, t2)
  t3 = Dense(512, activation="relu", name="dense3")(input_tensor3)
  t3 = Dense(512, activation="relu", name="dense33")(t3)
  model3 = Model(input_tensor3, t3)
  t4 = Dense(512, activation="relu", name="dense4")(input_tensor4)
  t4 = Dense(512, activation="relu", name="dense44")(t4)
  model4 = Model(input_tensor4, t4)
  
  input_tensor = Input(shape=(784,))
  t00 = Input(shape=(784,), name="input_00")
  t01 = Input(shape=(784,), name="input_01")
  t1 = model1(input_tensor)
  t2 = model2(input_tensor)
  t3 = model3(input_tensor)
  t4 = model4(input_tensor)
  output = Concatenate(axis=1)([t00, t01, t1, t2, t3, t4])
  output = Dense(num_classes)(output)
  output = Activation("softmax")(output)
  
  model = Model([t00, t01, input_tensor], output)

  opt = flexflow.keras.optimizers.SGD(learning_rate=0.01)
  model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy', 'sparse_categorical_crossentropy'])
  
  print(model.summary())

  model.fit([x_train, x_train, x_train], y_train, epochs=1)

if __name__ == "__main__":
  print("Functional API, mnist mlp concat with input")
  top_level_task()
  gc.collect()