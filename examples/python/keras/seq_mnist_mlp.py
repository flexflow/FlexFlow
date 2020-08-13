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
from flexflow.keras.layers import Flatten, Dense, Activation, Dropout
import flexflow.keras.optimizers
from flexflow.keras.datasets import mnist
from flexflow.keras.callbacks import Callback, VerifyMetrics, EpochVerifyMetrics
from flexflow.keras.initializers import GlorotUniform, Zeros

import flexflow.core as ff
import numpy as np
from accuracy import ModelAccuracy

def top_level_task():
  
  num_classes = 10
  
  (x_train, y_train), (x_test, y_test) = mnist.load_data()
  
  x_train = x_train.reshape(60000, 784)
  x_train = x_train.astype('float32')
  x_train /= 255
  y_train = y_train.astype('int32')
  y_train = np.reshape(y_train, (len(y_train), 1))
  print("shape: ", x_train.shape)
  
  model = Sequential()
  d1 = Dense(512, input_shape=(784,), kernel_initializer=GlorotUniform(123), bias_initializer=Zeros())
  model.add(d1)
  model.add(Activation('relu'))
  model.add(Dropout(0.2))
  model.add(Dense(512, activation="relu"))
  model.add(Dropout(0.2))
  model.add(Dense(num_classes))
  model.add(Activation("softmax"))

  opt = flexflow.keras.optimizers.SGD(learning_rate=0.01)
  model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy', 'sparse_categorical_crossentropy'])
  
  print(model.summary())

  model.fit(x_train, y_train, epochs=5, callbacks=[VerifyMetrics(ModelAccuracy.MNIST_MLP), EpochVerifyMetrics(ModelAccuracy.MNIST_MLP)])

if __name__ == "__main__":
  print("Sequential model, mnist mlp")
  top_level_task()
