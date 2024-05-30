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

from PIL import Image
  
def top_level_task():
  num_samples = 10000
  
  (x_train, y_train), (x_test, y_test) = cifar10.load_data(num_samples)

  full_input_np = np.zeros((num_samples, 3, 229, 229), dtype=np.float32)
  for i in range(0, num_samples):
    image = x_train[i, :, :, :]
    image = image.transpose(1, 2, 0)
    pil_image = Image.fromarray(image)
    pil_image = pil_image.resize((229,229), Image.Resampling.NEAREST)
    image = np.array(pil_image, dtype=np.float32)
    image = image.transpose(2, 0, 1)
    full_input_np[i, :, :, :] = image
    if (i == 0):
      print(image)
  
  full_input_np /= 255    
  y_train = y_train.astype('int32')
  full_label_np = y_train
  
  input_tensor = Input(shape=(3, 229, 229), dtype="float32")
  
  output = Conv2D(filters=64, input_shape=(3,229,229), kernel_size=(11,11), strides=(4,4), padding=(2,2), activation="relu")(input_tensor)
  output = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding="valid")(output)
  output = Conv2D(filters=192, kernel_size=(5,5), strides=(1,1), padding=(2,2), activation="relu")(output)
  output = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding="valid")(output)
  output = Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding=(1,1), activation="relu")(output)
  output = Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding=(1,1), activation="relu")(output)
  output = Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding=(1,1), activation="relu")(output)
  output = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding="valid")(output)
  output = Flatten()(output)
  output = Dense(4096, activation="relu")(output)
  output = Dense(4096, activation="relu")(output)
  output = Dense(10)(output)
  output = Activation("softmax")(output)
  
  model = Model(input_tensor, output)
  
  opt = flexflow.keras.optimizers.SGD(learning_rate=0.01)
  model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy', 'sparse_categorical_crossentropy'])
  print(model.summary())
  
  model.fit(full_input_np, full_label_np, epochs=160, callbacks=[VerifyMetrics(ModelAccuracy.CIFAR10_ALEXNET), EpochVerifyMetrics(ModelAccuracy.CIFAR10_ALEXNET)])

if __name__ == "__main__":
  print("Functional API, cifar10 alexnet")
  configs = ff.get_configs()
  ff.init_flexflow_runtime(configs)
  top_level_task()
  gc.collect()
