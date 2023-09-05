# Copyright 2023 CMU, Facebook, LANL, MIT, NVIDIA, and Stanford (alphabetical)
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
from flexflow.keras.layers import Input, Flatten, Dense, Activation, Conv2D, MaxPooling2D, Concatenate, concatenate
import flexflow.keras.optimizers
from flexflow.keras.datasets import mnist
from flexflow.keras.datasets import cifar10
from flexflow.keras import losses
from flexflow.keras import metrics
from flexflow.keras.callbacks import Callback, VerifyMetrics, EpochVerifyMetrics
from accuracy import ModelAccuracy

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
  
  #teacher
  
  input_tensor1 = Input(shape=(784,), dtype="float32")
  
  d1 = Dense(512, input_shape=(784,), activation="relu")
  d2 = Dense(512, activation="relu")
  d3 = Dense(num_classes)
  
  output = d1(input_tensor1)
  output = d2(output)
  output = d3(output)
  output = Activation("softmax")(output)
  
  teacher_model = Model(input_tensor1, output)

  opt = flexflow.keras.optimizers.SGD(learning_rate=0.01)
  teacher_model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy', 'sparse_categorical_crossentropy'])

  teacher_model.fit(x_train, y_train, epochs=10)
  
  d1_kernel, d1_bias = d1.get_weights(teacher_model.ffmodel)
  d2_kernel, d2_bias = d2.get_weights(teacher_model.ffmodel)
  d3_kernel, d3_bias = d3.get_weights(teacher_model.ffmodel)
  
  # student
  
  input_tensor2 = Input(shape=(784,), dtype="float32")
  
  sd1_1 = Dense(512, input_shape=(784,), activation="relu")
  sd2 = Dense(512, activation="relu")
  sd3 = Dense(num_classes)
  
  output = sd1_1(input_tensor2)
  output = sd2(output)
  output = sd3(output)
  output = Activation("softmax")(output)

  student_model = Model(input_tensor2, output)

  opt = flexflow.keras.optimizers.SGD(learning_rate=0.01)
  student_model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy', 'sparse_categorical_crossentropy'])
  
  sd1_1.set_weights(student_model.ffmodel, d1_kernel, d1_bias)
  sd2.set_weights(student_model.ffmodel, d2_kernel, d2_bias)
  sd3.set_weights(student_model.ffmodel, d3_kernel, d3_bias)

  student_model.fit(x_train, y_train, epochs=160, callbacks=[VerifyMetrics(ModelAccuracy.MNIST_MLP), EpochVerifyMetrics(ModelAccuracy.MNIST_MLP)])


if __name__ == "__main__":
  print("Functional API, mnist mlp teach student")
  configs = ff.get_configs()
  ff.init_flexflow_runtime(configs)
  top_level_task()
  gc.collect()
