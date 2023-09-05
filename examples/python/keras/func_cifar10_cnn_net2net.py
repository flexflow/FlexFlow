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
  
  num_samples = 10000
  
  (x_train, y_train), (x_test, y_test) = cifar10.load_data(num_samples)
  
  x_train = x_train.astype('float32')
  x_train /= 255
  y_train = y_train.astype('int32')
  print("shape: ", x_train.shape)
  
  #teacher
  input_tensor1 = Input(shape=(3, 32, 32), dtype="float32")

  c1 = Conv2D(filters=32, input_shape=(3,32,32), kernel_size=(3,3), strides=(1,1), padding="same", activation="relu")
  c2 = Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding=(1,1), activation="relu")
  c3 = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding=(1,1), activation="relu")
  c4 = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding=(1,1), activation="relu")
  d1 = Dense(512, activation="relu")
  d2 = Dense(num_classes)

  output_tensor = c1(input_tensor1)
  output_tensor = c2(output_tensor)
  output_tensor = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="same")(output_tensor)
  output_tensor = c3(output_tensor)
  output_tensor = c4(output_tensor)
  output_tensor = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="valid")(output_tensor)
  output_tensor = Flatten()(output_tensor)
  output_tensor = d1(output_tensor)
  output_tensor = d2(output_tensor)
  output_tensor = Activation("softmax")(output_tensor)

  teacher_model = Model(input_tensor1, output_tensor)

  opt = flexflow.keras.optimizers.SGD(learning_rate=0.01)
  teacher_model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy', 'sparse_categorical_crossentropy'])

  teacher_model.fit(x_train, y_train, epochs=10)

  c1_kernel, c1_bias = c1.get_weights(teacher_model.ffmodel)
  c2_kernel, c2_bias = c2.get_weights(teacher_model.ffmodel)
  c3_kernel, c3_bias = c3.get_weights(teacher_model.ffmodel)
  c4_kernel, c4_bias = c4.get_weights(teacher_model.ffmodel)
  d1_kernel, d1_bias = d1.get_weights(teacher_model.ffmodel)
  d2_kernel, d2_bias = d2.get_weights(teacher_model.ffmodel)
  #d2_kernel *= 0

  c2_kernel_new = np.concatenate((c2_kernel, c2_kernel), axis=1)
  print(c2_kernel.shape, c2_kernel_new.shape, c2_bias.shape)
  
  #student model
  input_tensor2 = Input(shape=(3, 32, 32), dtype="float32")

  sc1_1 = Conv2D(filters=32, input_shape=(3,32,32), kernel_size=(3,3), strides=(1,1), padding="same", activation="relu")
  sc1_2 = Conv2D(filters=32, input_shape=(3,32,32), kernel_size=(3,3), strides=(1,1), padding="same", activation="relu")
  sc2 = Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding=(1,1), activation="relu")
  sc3 = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding=(1,1), activation="relu")
  sc4 = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding=(1,1), activation="relu")
  sd1 = Dense(512, activation="relu")
  sd2 = Dense(num_classes)

  t1 = sc1_1(input_tensor2)
  t2 = sc1_2(input_tensor2)
  output_tensor = Concatenate(axis=1)([t1, t2])
  output_tensor = sc2(output_tensor)
  output_tensor = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="same")(output_tensor)
  output_tensor = sc3(output_tensor)
  output_tensor = sc4(output_tensor)
  output_tensor = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="valid")(output_tensor)
  output_tensor = Flatten()(output_tensor)
  output_tensor = sd1(output_tensor)
  output_tensor = sd2(output_tensor)
  output_tensor = Activation("softmax")(output_tensor)

  student_model = Model(input_tensor2, output_tensor)

  opt = flexflow.keras.optimizers.SGD(learning_rate=0.01)
  student_model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy', 'sparse_categorical_crossentropy'])

  sc1_1.set_weights(student_model.ffmodel, c1_kernel, c1_bias)
  sc1_2.set_weights(student_model.ffmodel, c1_kernel, c1_bias)
  sc2.set_weights(student_model.ffmodel, c2_kernel_new, c2_bias)
  sc3.set_weights(student_model.ffmodel, c3_kernel, c3_bias)
  sc4.set_weights(student_model.ffmodel, c4_kernel, c4_bias)
  sd1.set_weights(student_model.ffmodel, d1_kernel, d1_bias)
  sd2.set_weights(student_model.ffmodel, d2_kernel, d2_bias)

  student_model.fit(x_train, y_train, epochs=160, callbacks=[VerifyMetrics(ModelAccuracy.CIFAR10_CNN), EpochVerifyMetrics(ModelAccuracy.CIFAR10_CNN)])

if __name__ == "__main__":
  print("Functional API, cifarf10 cnn teach student")
  configs = ff.get_configs()
  ff.init_flexflow_runtime(configs)
  top_level_task()
  gc.collect()
