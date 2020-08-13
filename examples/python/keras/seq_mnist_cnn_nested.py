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

from flexflow.keras.models import Sequential, Model
from flexflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, AveragePooling2D, Input
import flexflow.keras.optimizers
from flexflow.keras.datasets import mnist
from flexflow.keras.callbacks import Callback, VerifyMetrics, EpochVerifyMetrics

import flexflow.core as ff
import numpy as np
from accuracy import ModelAccuracy

def top_level_task():
  
  num_classes = 10

  img_rows, img_cols = 28, 28
  
  (x_train, y_train), (x_test, y_test) = mnist.load_data()
  x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
  
  x_train = x_train.astype('float32')
  x_train /= 255
  y_train = y_train.astype('int32')
  y_train = np.reshape(y_train, (len(y_train), 1))
  print("shape: ", x_train.shape, x_train.__array_interface__["strides"])
  
  layers = [Conv2D(filters=32, input_shape=(1,28,28), kernel_size=(3,3), strides=(1,1), padding=(1,1), activation="relu"),
            Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding=(1,1), activation="relu"),
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="valid"),
            Flatten()]
  model1 = Sequential(layers)
  
  input_tensor = Input(shape=(12544,), dtype="float32")
  
  output = Dense(512, input_shape=(12544,), activation="relu")(input_tensor)
  output = Dense(num_classes)(output)
  output = Activation("softmax")(output)
  
  model2 = Model(input_tensor, output)
  
  model = Sequential()
  model.add(model1)
  model.add(model2)
  
  print(model.summary())

  opt = flexflow.keras.optimizers.SGD(learning_rate=0.01)
  model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy', 'sparse_categorical_crossentropy'])
  
  print(model.summary())

  model.fit(x_train, y_train, epochs=5, callbacks=[VerifyMetrics(ModelAccuracy.MNIST_CNN), EpochVerifyMetrics(ModelAccuracy.MNIST_CNN)])

if __name__ == "__main__":
  print("Sequential model, mnist cnn nested model")
  top_level_task()
