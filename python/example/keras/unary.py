# Copyright 2020 Stanford University, Los Alamos National Laboratory
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

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
  model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy', 'sparse_categorical_crossentropy'])
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