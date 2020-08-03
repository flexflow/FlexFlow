from flexflow.keras.models import Model, Sequential
from flexflow.keras.layers import Add, Subtract, Flatten, Dense, Activation, Conv2D, MaxPooling2D, Concatenate, add, subtract, Input
import flexflow.keras.optimizers
from flexflow.keras.datasets import mnist
from flexflow.keras.datasets import cifar10

import flexflow.core as ff
import numpy as np
import argparse
import gc

def add_test():
  input1 = Input(shape=(16, ), dtype="float32")
  x1 = Dense(8, activation='relu')(input1)
  input2 = Input(shape=(32,), dtype="float32")
  x2 = Dense(8, activation='relu')(input2)
  subtracted = Add()([x1, x2])

  out = Dense(4)(subtracted)
  model = Model([input1, input2], out)

  opt = flexflow.keras.optimizers.SGD(learning_rate=0.01)
  model.compile(optimizer=opt)
  print(model.summary())
  model.ffmodel.init_layers()
  
def subtract_test():
  input1 = Input(shape=(16, ), dtype="float32")
  x1 = Dense(8, activation='relu')(input1)
  input2 = Input(shape=(32, ), dtype="float32")
  x2 = Dense(8, activation='relu')(input2)
  subtracted = subtract([x1, x2])

  out = Dense(4)(subtracted)
  model = Model([input1, input2], out)

  opt = flexflow.keras.optimizers.SGD(learning_rate=0.01)
  model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy', 'sparse_categorical_crossentropy'])
  print(model.summary())
  model.ffmodel.init_layers()

def top_level_task():
  
  add_test()
  subtract_test()


if __name__ == "__main__":
  print("alexnet keras")
  top_level_task()