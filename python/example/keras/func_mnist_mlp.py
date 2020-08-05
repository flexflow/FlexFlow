from flexflow.keras.models import Model, Sequential
from flexflow.keras.layers import Input, Flatten, Dense, Activation, Conv2D, MaxPooling2D, Concatenate, concatenate
import flexflow.keras.optimizers
from flexflow.keras.datasets import mnist
from flexflow.keras.datasets import cifar10
from flexflow.keras import losses
from flexflow.keras import metrics
from flexflow.keras.callbacks import Callback, VerifyMetrics
from example.accuracy import ModelAccuracy

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
  
  input_tensor = Input(shape=(784,))
  
  output = Dense(512, input_shape=(784,), activation="relu")(input_tensor)
  output2 = Dense(512, activation="relu")(output)
  output3 = Dense(num_classes)(output2)
  output4 = Activation("softmax")(output3)
  
  model = Model(input_tensor, output4)

  opt = flexflow.keras.optimizers.SGD(learning_rate=0.01)
  model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy', metrics.SparseCategoricalCrossentropy()])

  model.fit(x_train, y_train, batch_size=64, epochs=1, callbacks=[VerifyMetrics(ModelAccuracy.MNIST_MLP)])

if __name__ == "__main__":
  print("Functional API, mnist mlp")
  top_level_task()
  gc.collect()