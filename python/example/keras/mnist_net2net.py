from flexflow.keras.models import Sequential
from flexflow.keras.layers import Flatten, Dense, Activation, Conv2D, MaxPooling2D
import flexflow.keras.optimizers
from flexflow.keras.datasets import mnist

import flexflow.core as ff
import numpy as np

def create_teacher_model(num_classes, x_train, y_train):
  model = Sequential()
  model.add(Dense(512, input_shape=(784,), activation="relu"))
  model.add(Dense(512, activation="relu"))
  model.add(Dense(num_classes))
  model.add(Activation("softmax"))

  opt = flexflow.keras.optimizers.SGD(learning_rate=0.01)
  model.compile(optimizer=opt)

  model.fit(x_train, y_train, epochs=1)
  
  dense3 = model.get_layer(2)
  d3_kernel, d3_bias = dense3.get_weights(model.ffmodel)
  print(d3_bias)
  d3_kernel = np.reshape(d3_kernel, (d3_kernel.shape[1], d3_kernel.shape[0]))
  print(d3_kernel)
  return model
  
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
  model.compile(optimizer=opt)
  
  print(model.summary())

  model.fit(x_train, y_train, epochs=1)
  return model
  
def create_student_model(teacher_model, num_classes, x_train, y_train):
  dense1 = teacher_model.get_layer(0)
  d1_kernel, d1_bias = dense1.get_weights(teacher_model.ffmodel)
  print(d1_kernel.shape, d1_bias.shape)
  # print(d1_kernel)
  # print(d1_bias)
  dense2 = teacher_model.get_layer(1)
  d2_kernel, d2_bias = dense2.get_weights(teacher_model.ffmodel)
  
  dense3 = teacher_model.get_layer(2)
  d3_kernel, d3_bias = dense3.get_weights(teacher_model.ffmodel)
  
  model = Sequential()
  model.add(Dense(512, input_shape=(784,), activation="relu"))
  model.add(Dense(512, activation="relu"))
  model.add(Dense(num_classes))
  model.add(Activation("softmax"))

  opt = flexflow.keras.optimizers.SGD(learning_rate=0.01)
  model.compile(optimizer=opt)
  
  dense1s = model.get_layer(0)
  dense2s = model.get_layer(1)
  dense3s = model.get_layer(2)
  
  dense1s.set_weights(model.ffmodel, d1_kernel, d1_bias)
  dense2s.set_weights(model.ffmodel, d2_kernel, d2_bias)
  dense3s.set_weights(model.ffmodel, d3_kernel, d3_bias)
  
  d3_kernel, d3_bias = dense3s.get_weights(model.ffmodel)
  print(d3_kernel)
  print(d3_bias)
  

  model.fit(x_train, y_train, epochs=1)
  
def create_student_model_cnn(teacher_model, num_classes, x_train, y_train):
  conv1 = teacher_model.get_layer(0)
  c1_kernel, c1_bias = conv1.get_weights(teacher_model.ffmodel)
  print(c1_kernel.shape, c1_bias.shape)

  conv2 = teacher_model.get_layer(1)
  c2_kernel, c2_bias = conv2.get_weights(teacher_model.ffmodel)
  
  dense1 = teacher_model.get_layer(4)
  d1_kernel, d1_bias = dense1.get_weights(teacher_model.ffmodel)
  
  dense2 = teacher_model.get_layer(5)
  d2_kernel, d2_bias = dense2.get_weights(teacher_model.ffmodel)
  
  model = Sequential()
  model.add(Conv2D(filters=32, input_shape=(1,28,28), kernel_size=(3,3), strides=(1,1), padding=(1,1), activation="relu"))
  model.add(Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding=(1,1), activation="relu"))
  model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="valid"))
  model.add(Flatten())
  model.add(Dense(128, activation="relu"))
  model.add(Dense(num_classes))
  model.add(Activation("softmax"))

  opt = flexflow.keras.optimizers.SGD(learning_rate=0.01)
  model.compile(optimizer=opt)
  
  conv1s = model.get_layer(0)
  conv2s = model.get_layer(1)
  dense1s = model.get_layer(4)
  dense2s = model.get_layer(5)
  
  conv1s.set_weights(model.ffmodel, c1_kernel, c1_bias)
  conv2s.set_weights(model.ffmodel, c2_kernel, c2_bias)
  dense1s.set_weights(model.ffmodel, d1_kernel, d1_bias)
  dense2s.set_weights(model.ffmodel, d2_kernel, d2_bias)
  
  print(model.summary())
  

  model.fit(x_train, y_train, epochs=1)
  
def mlp():
  num_classes = 10

  (x_train, y_train), (x_test, y_test) = mnist.load_data()

  x_train = x_train.reshape(60000, 784)
  x_train = x_train.astype('float32')
  x_train /= 255
  y_train = y_train.astype('int32')
  y_train = np.reshape(y_train, (len(y_train), 1))
  #y_train = np.random.randint(1, 9, size=(len(y_train),1), dtype='int32')
  print("shape: ", x_train.shape)

  teacher_model = create_teacher_model(num_classes, x_train, y_train)

  create_student_model(teacher_model, num_classes, x_train, y_train)
   
def cnn():
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
  

def top_level_task():
  cnn()
  #mlp()

if __name__ == "__main__":
  print("alexnet keras")
  top_level_task()