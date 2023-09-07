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

from flexflow.keras.models import Sequential
from flexflow.keras.layers import Flatten, Dense, Activation, Conv2D, MaxPooling2D
import flexflow.keras.optimizers
from flexflow.keras.datasets import mnist
from flexflow.keras.callbacks import Callback, VerifyMetrics, EpochVerifyMetrics

import flexflow.core as ff
import numpy as np
from accuracy import ModelAccuracy
  
def create_teacher_model_cnn(num_classes, x_train, y_train):
  model = Sequential()
  model.add(Conv2D(filters=32, input_shape=(1,28,28), kernel_size=(3,3), strides=(1,1), padding=(1,1), activation="relu"))
  model.add(Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding=(1,1), activation="relu"))
  model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="valid"))
  model.add(Flatten())
  model.add(Dense(128, activation="relu"))
  model.add(Dense(num_classes))
  model.add(Activation("softmax"))

  opt = flexflow.keras.optimizers.SGD(learning_rate=0.01)
  model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy', 'sparse_categorical_crossentropy'])
  
  print(model.summary())

  model.fit(x_train, y_train, epochs=5)
  return model
  
def create_student_model_cnn(teacher_model, num_classes, x_train, y_train):
  conv1 = teacher_model.get_layer(index=0)
  c1_kernel, c1_bias = conv1.get_weights(teacher_model.ffmodel)
  print(c1_kernel.shape, c1_bias.shape)

  conv2 = teacher_model.get_layer(index=1)
  c2_kernel, c2_bias = conv2.get_weights(teacher_model.ffmodel)
  
  dense1 = teacher_model.get_layer(index=4)
  d1_kernel, d1_bias = dense1.get_weights(teacher_model.ffmodel)
  
  dense2 = teacher_model.get_layer(index=5)
  d2_kernel, d2_bias = dense2.get_weights(teacher_model.ffmodel)
  
  model = Sequential()
  model.add(Conv2D(filters=32, input_shape=(1,28,28), kernel_size=(3,3), strides=(1,1), padding=(1,1), activation="relu"))
  model.add(Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding=(1,1), activation="relu"))
  model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="valid"))
  model.add(Flatten())
  model.add(Dense(128, activation="relu", name="dense1"))
  model.add(Dense(num_classes))
  model.add(Activation("softmax"))

  opt = flexflow.keras.optimizers.SGD(learning_rate=0.01)
  model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy', 'sparse_categorical_crossentropy'])
  
  conv1s = model.get_layer(index=0)
  conv2s = model.get_layer(index=1)
  dense1s = model.get_layer(name="dense1")
  dense2s = model.get_layer(index=5)
  
  conv1s.set_weights(model.ffmodel, c1_kernel, c1_bias)
  conv2s.set_weights(model.ffmodel, c2_kernel, c2_bias)
  dense1s.set_weights(model.ffmodel, d1_kernel, d1_bias)
  dense2s.set_weights(model.ffmodel, d2_kernel, d2_bias)
  
  print(model.summary())
  
  model.fit(x_train, y_train, epochs=5, callbacks=[VerifyMetrics(ModelAccuracy.MNIST_CNN), EpochVerifyMetrics(ModelAccuracy.MNIST_CNN)])
   
def top_level_task():
  num_classes = 10

  img_rows, img_cols = 28, 28
  
  (x_train, y_train), (x_test, y_test) = mnist.load_data()
  x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
  
  x_train = x_train.astype('float32')
  x_train /= 255
  y_train = y_train.astype('int32')
  y_train = np.reshape(y_train, (len(y_train), 1))
  
  teacher_model = create_teacher_model_cnn(num_classes, x_train, y_train)

  create_student_model_cnn(teacher_model, num_classes, x_train, y_train)


if __name__ == "__main__":
  print("Sequential model, mnist mlp teacher student")
  configs = ff.get_configs()
  ff.init_flexflow_runtime(configs)
  top_level_task()
