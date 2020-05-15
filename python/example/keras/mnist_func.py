from flexflow.keras.models import Model, Input
from flexflow.keras.layers import Flatten, Dense, Activation, Conv2D, MaxPooling2D, Concatenate
import flexflow.keras.optimizers
from flexflow.keras.datasets import mnist
from flexflow.keras.datasets import cifar10

import flexflow.core as ff
import numpy as np

from PIL import Image
  
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
  
  input_tensor = Input(batch_shape=[0, 784], dtype="float32")
  
  output = Dense(512, input_shape=(784,), activation="relu")(input_tensor)
  output2 = Dense(512, activation="relu")(output)
  output3 = Dense(num_classes)(output2)
  output4 = Activation("softmax")(output3)
  
  model = Model(input_tensor, output4)

  opt = flexflow.keras.optimizers.SGD(learning_rate=0.01)
  model.compile(optimizer=opt)

  model.fit(x_train, y_train, epochs=1)
  
  # del output
  # del output2
  # del output3

def mlp_concat():
  num_classes = 10

  (x_train, y_train), (x_test, y_test) = mnist.load_data()

  x_train = x_train.reshape(60000, 784)
  x_train = x_train.astype('float32')
  x_train /= 255
  y_train = y_train.astype('int32')
  y_train = np.reshape(y_train, (len(y_train), 1))
  #y_train = np.random.randint(1, 9, size=(len(y_train),1), dtype='int32')
  print("shape: ", x_train.shape)
  
  input_tensor = Input(batch_shape=[0, 784], dtype="float32")
  
  t1 = Dense(512, input_shape=(784,), activation="relu")(input_tensor)
  t2 = Dense(512, input_shape=(784,), activation="relu")(input_tensor)
  t3 = Dense(512, input_shape=(784,), activation="relu")(input_tensor)
  t4 = Dense(512, input_shape=(784,), activation="relu")(input_tensor)
  output = Concatenate(axis=1)([t1, t2, t3, t4])
  output2 = Dense(512, activation="relu")(output)
  output3 = Dense(num_classes)(output2)
  output4 = Activation("softmax")(output3)
  
  model = Model(input_tensor, output4)
  
  print(model.summary())

  opt = flexflow.keras.optimizers.SGD(learning_rate=0.01)
  model.compile(optimizer=opt)

  model.fit(x_train, y_train, epochs=1)
   
def cnn():
  num_classes = 10

  img_rows, img_cols = 28, 28
  
  (x_train, y_train), (x_test, y_test) = mnist.load_data()
  x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
  
  x_train = x_train.astype('float32')
  x_train /= 255
  y_train = y_train.astype('int32')
  y_train = np.reshape(y_train, (len(y_train), 1))
  
  input_tensor = Input(batch_shape=[0, 1, 28, 28], dtype="float32")
  
  output = Conv2D(filters=32, input_shape=(1,28,28), kernel_size=(3,3), strides=(1,1), padding=(1,1), activation="relu")(input_tensor)
  output = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding=(1,1), activation="relu")(output)
  output = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="valid")(output)
  output = Flatten()(output)
  output = Dense(128, activation="relu")(output)
  output = Dense(num_classes)(output)
  output = Activation("softmax")(output)

  model = Model(input_tensor, output)
  
  opt = flexflow.keras.optimizers.SGD(learning_rate=0.01)
  model.compile(optimizer=opt)
  
  model.fit(x_train, y_train, epochs=1)
  
def cnn_concat():
  num_classes = 10

  img_rows, img_cols = 28, 28
  
  (x_train, y_train), (x_test, y_test) = mnist.load_data()
  x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
  
  x_train = x_train.astype('float32')
  x_train /= 255
  y_train = y_train.astype('int32')
  y_train = np.reshape(y_train, (len(y_train), 1))
  
  input_tensor = Input(batch_shape=[0, 1, 28, 28], dtype="float32")
  
  t1 = Conv2D(filters=32, input_shape=(1,28,28), kernel_size=(3,3), strides=(1,1), padding=(1,1), activation="relu")(input_tensor)
  t2 = Conv2D(filters=32, input_shape=(1,28,28), kernel_size=(3,3), strides=(1,1), padding=(1,1), activation="relu")(input_tensor)
  output = Concatenate(axis=1)([t1, t2])
  output = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding=(1,1), activation="relu")(output)
  output = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="valid")(output)
  output = Flatten()(output)
  output = Dense(128, activation="relu")(output)
  output = Dense(num_classes)(output)
  output = Activation("softmax")(output)

  model = Model(input_tensor, output)
  
  opt = flexflow.keras.optimizers.SGD(learning_rate=0.01)
  model.compile(optimizer=opt)
  
  model.fit(x_train, y_train, epochs=1)
  
def cifar_cnn():
  num_classes = 10
  
  num_samples = 10000
  
  (x_train, y_train), (x_test, y_test) = cifar10.load_data(num_samples)
  
  x_train = x_train.astype('float32')
  x_train /= 255
  #x_train *= 0
  #y_train = np.random.randint(1, 9, size=(num_samples,1), dtype='int32')
  y_train = y_train.astype('int32')
  print("shape: ", x_train.shape)
  
  input_tensor1 = Input(batch_shape=[0, 3, 32, 32], dtype="float32")
  
  output_tensor = Conv2D(filters=32, input_shape=(3,32,32), kernel_size=(3,3), strides=(1,1), padding=(1,1), activation="relu")(input_tensor1)
  output_tensor = Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding=(1,1), activation="relu")(output_tensor)
  output_tensor = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="valid")(output_tensor)
  output_tensor = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding=(1,1), activation="relu")(output_tensor)
  output_tensor = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding=(1,1), activation="relu")(output_tensor)
  output_tensor = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="valid")(output_tensor)
  output_tensor = Flatten()(output_tensor)
  output_tensor = Dense(512, activation="relu")(output_tensor)
  output_tensor = Dense(num_classes)(output_tensor)
  output_tensor = Activation("softmax")(output_tensor)

  model = Model(input_tensor1, output_tensor)
  
  print(model.summary())
  
  opt = flexflow.keras.optimizers.SGD(learning_rate=0.01)
  model.compile(optimizer=opt)

  model.fit(x_train, y_train, epochs=1)


def cifar_cnn_sub(input_tensor):
  t1 = Conv2D(filters=32, input_shape=(3,32,32), kernel_size=(3,3), strides=(1,1), padding=(1,1), activation="relu")(input_tensor)
  ot1 = Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding=(1,1), activation="relu")(t1)
  return ot1
    
def cifar_cnn_concat():
  num_classes = 10
  
  num_samples = 10000
  
  (x_train, y_train), (x_test, y_test) = cifar10.load_data(num_samples)
  
  x_train = x_train.astype('float32')
  x_train /= 255
  #x_train *= 0
  #y_train = np.random.randint(1, 9, size=(num_samples,1), dtype='int32')
  y_train = y_train.astype('int32')
  print("shape: ", x_train.shape)
  
  input_tensor1 = Input(batch_shape=[0, 3, 32, 32], dtype="float32")
  input_tensor2 = Input(batch_shape=[0, 3, 32, 32], dtype="float32")

  ot1 = cifar_cnn_sub(input_tensor1)
  ot2 = cifar_cnn_sub(input_tensor1)
  ot3 = cifar_cnn_sub(input_tensor1)
  output_tensor = Concatenate(axis=1)([ot1, ot2, ot3])
  # output_tensor = Conv2D(filters=32, input_shape=(3,32,32), kernel_size=(3,3), strides=(1,1), padding=(1,1), activation="relu")(input_tensor)
  # output_tensor = Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding=(1,1), activation="relu")(output_tensor)
  output_tensor = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="valid")(output_tensor)
  o1 = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding=(1,1), activation="relu")(output_tensor)
  o2 = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding=(1,1), activation="relu")(output_tensor)
  output_tensor = Concatenate(axis=1)([o1, o2])
  output_tensor = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding=(1,1), activation="relu")(output_tensor)
  output_tensor = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="valid")(output_tensor)
  output_tensor = Flatten()(output_tensor)
  output_tensor = Dense(512, activation="relu")(output_tensor)
  output_tensor = Dense(num_classes)(output_tensor)
  output_tensor = Activation("softmax")(output_tensor)

  model = Model([input_tensor1, input_tensor2], output_tensor)
  
  print(model.summary())
  
  opt = flexflow.keras.optimizers.SGD(learning_rate=0.01)
  model.compile(optimizer=opt)

  model.fit([x_train, x_train], y_train, epochs=1)
  
def cifar_alexnet_concat():
  
  num_samples = 10000
  
  (x_train, y_train), (x_test, y_test) = cifar10.load_data(num_samples)

  full_input_np = np.zeros((num_samples, 3, 229, 229), dtype=np.float32)
  for i in range(0, num_samples):
    image = x_train[i, :, :, :]
    image = image.transpose(1, 2, 0)
    pil_image = Image.fromarray(image)
    pil_image = pil_image.resize((229,229), Image.NEAREST)
    image = np.array(pil_image, dtype=np.float32)
    image = image.transpose(2, 0, 1)
    full_input_np[i, :, :, :] = image
    if (i == 0):
      print(image)
  
  full_input_np /= 255    
  y_train = y_train.astype('int32')
  full_label_np = y_train
  
  input_tensor = Input(batch_shape=[0, 3, 229, 229], dtype="float32")
  
  t1 = Conv2D(filters=64, input_shape=(3,229,229), kernel_size=(11,11), strides=(4,4), padding=(2,2))(input_tensor)
  t2 = Conv2D(filters=64, input_shape=(3,229,229), kernel_size=(11,11), strides=(4,4), padding=(2,2))(input_tensor)
  output = Concatenate(axis=1)([t1, t2])
  output = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding="valid")(output)
  output = Conv2D(filters=192, kernel_size=(5,5), strides=(1,1), padding=(2,2))(output)
  output = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding="valid")(output)
  output = Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding=(1,1))(output)
  output = Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding=(1,1))(output)
  output = Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding=(1,1))(output)
  output = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding="valid")(output)
  output = Flatten()(output)
  output = Dense(4096, activation="relu")(output)
  output = Dense(4096, activation="relu")(output)
  output = Dense(10)(output)
  output = Activation("softmax")(output)
  
  model = Model(input_tensor, output)
  
  opt = flexflow.keras.optimizers.SGD(learning_rate=0.001)
  model.compile(optimizer=opt)
  
  model.fit(full_input_np, full_label_np, epochs=1)
  

def top_level_task():
  
  #cnn()
  #cnn_concat()
  #cifar_cnn()
  cifar_cnn_concat()
  #cifar_alexnet_concat()
  #mlp()
  #mlp_concat()

if __name__ == "__main__":
  print("alexnet keras")
  top_level_task()