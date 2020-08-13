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
from flexflow.keras.layers import Flatten, Dense, Activation, Input
import flexflow.keras.optimizers
from flexflow.keras.datasets import reuters
from flexflow.keras.preprocessing.text import Tokenizer
from flexflow.keras.callbacks import Callback, VerifyMetrics

import numpy as np
from accuracy import ModelAccuracy

def top_level_task():
  
  max_words = 1000
  epochs = 5
  
  print('Loading data...')
  (x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=max_words,
                                                           test_split=0.2)
  print(len(x_train), 'train sequences')
  print(len(x_test), 'test sequences')

  num_classes = np.max(y_train) + 1
  print(num_classes, 'classes')
  
  print('Vectorizing sequence data...')
  tokenizer = Tokenizer(num_words=max_words)
  x_train = tokenizer.sequences_to_matrix(x_train, mode='binary')
  x_test = tokenizer.sequences_to_matrix(x_test, mode='binary')
  x_train = x_train.astype('float32')
  print('x_train shape:', x_train.shape)
  print('x_test shape:', x_test.shape)

  y_train = y_train.astype('int32')
  y_train = np.reshape(y_train, (len(y_train), 1))
  print('y_train shape:', y_train.shape)
  
  model = Sequential()
  model.add(Input(shape=(max_words,)))
  model.add(Dense(512, activation="relu"))
  model.add(Dense(num_classes))
  model.add(Activation("softmax"))

  opt = flexflow.keras.optimizers.Adam(learning_rate=0.01)
  model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy', 'sparse_categorical_crossentropy'])
  print(model.summary())

  model.fit(x_train, y_train, epochs=epochs, callbacks=[VerifyMetrics(ModelAccuracy.REUTERS_MLP)])

if __name__ == "__main__":
  print("Sequential model, reuters mlp")
  top_level_task()
