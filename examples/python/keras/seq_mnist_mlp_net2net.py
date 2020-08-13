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

from flexflow.keras.models import Sequential
from flexflow.keras.layers import Flatten, Dense, Activation, Conv2D, MaxPooling2D
import flexflow.keras.optimizers
from flexflow.keras.datasets import mnist
from flexflow.keras.callbacks import Callback, VerifyMetrics, EpochVerifyMetrics

import flexflow.core as ff
import numpy as np
from accuracy import ModelAccuracy

def create_teacher_model_mlp(num_classes, x_train, y_train):
  model = Sequential()
  model.add(Dense(512, input_shape=(784,), activation="relu"))
  model.add(Dense(512, activation="relu"))
  model.add(Dense(num_classes))
  model.add(Activation("softmax"))

  opt = flexflow.keras.optimizers.SGD(learning_rate=0.01)
  model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy', 'sparse_categorical_crossentropy'])

  model.fit(x_train, y_train, epochs=1)
  
  dense3 = model.get_layer(index=2)
  d3_kernel, d3_bias = dense3.get_weights(model.ffmodel)
  print(d3_bias)
  d3_kernel = np.reshape(d3_kernel, (d3_kernel.shape[1], d3_kernel.shape[0]))
  print(d3_kernel)
  return model
  
def create_student_model_mlp(teacher_model, num_classes, x_train, y_train):
  dense1 = teacher_model.get_layer(index=0)
  d1_kernel, d1_bias = dense1.get_weights(teacher_model.ffmodel)
  print(d1_kernel.shape, d1_bias.shape)
  dense2 = teacher_model.get_layer(index=1)
  d2_kernel, d2_bias = dense2.get_weights(teacher_model.ffmodel)
  
  dense3 = teacher_model.get_layer(index=2)
  d3_kernel, d3_bias = dense3.get_weights(teacher_model.ffmodel)
  
  model = Sequential()
  model.add(Dense(512, input_shape=(784,), activation="relu"))
  model.add(Dense(512, activation="relu"))
  model.add(Dense(num_classes))
  model.add(Activation("softmax"))

  opt = flexflow.keras.optimizers.SGD(learning_rate=0.01)
  model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy', 'sparse_categorical_crossentropy'])
  
  dense1s = model.get_layer(index=0)
  dense2s = model.get_layer(index=1)
  dense3s = model.get_layer(index=2)
  
  dense1s.set_weights(model.ffmodel, d1_kernel, d1_bias)
  dense2s.set_weights(model.ffmodel, d2_kernel, d2_bias)
  dense3s.set_weights(model.ffmodel, d3_kernel, d3_bias)
  
  d3_kernel, d3_bias = dense3s.get_weights(model.ffmodel)
  print(d3_kernel)
  print(d3_bias)

  model.fit(x_train, y_train, epochs=5, callbacks=[VerifyMetrics(ModelAccuracy.MNIST_MLP), EpochVerifyMetrics(ModelAccuracy.MNIST_MLP)])
  
def top_level_task():
  num_classes = 10

  (x_train, y_train), (x_test, y_test) = mnist.load_data()

  x_train = x_train.reshape(60000, 784)
  x_train = x_train.astype('float32')
  x_train /= 255
  y_train = y_train.astype('int32')
  y_train = np.reshape(y_train, (len(y_train), 1))
  print("shape: ", x_train.shape)

  teacher_model = create_teacher_model_mlp(num_classes, x_train, y_train)

  create_student_model_mlp(teacher_model, num_classes, x_train, y_train)

if __name__ == "__main__":
  print("Sequential model, mnist mlp teacher student")
  top_level_task()
