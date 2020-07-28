from flexflow.keras.models import Model, Sequential
from flexflow.keras.layers import Input, Flatten, Dense, Activation, Conv2D, MaxPooling2D, Concatenate, concatenate
import flexflow.keras.optimizers
from flexflow.keras.datasets import mnist
from flexflow.keras.datasets import cifar10

import flexflow.core as ff
import numpy as np
import argparse
import gc

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
  
  input_tensor = Input(shape=(784,))
  
  output = Dense(512, input_shape=(784,), activation="relu")(input_tensor)
  output2 = Dense(512, activation="relu")(output)
  output3 = Dense(num_classes)(output2)
  output4 = Activation("softmax")(output3)
  
  model = Model(input_tensor, output4)

  opt = flexflow.keras.optimizers.SGD(learning_rate=0.01)
  model.compile(optimizer=opt)

  model.fit(x_train, y_train, batch_size=64, epochs=1)
  
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
  
  input_tensor1 = Input(shape=(256,))
  input_tensor2 = Input(shape=(256,))
  input_tensor3 = Input(shape=(256,))
  input_tensor4 = Input(shape=(256,))
  
  t1 = Dense(512, activation="relu", name="dense1")(input_tensor1)
  t1 = Dense(512, activation="relu", name="dense12")(t1)
  model1 = Model(input_tensor1, t1)
  t2 = Dense(512, activation="relu", name="dense2")(input_tensor2)
  t2 = Dense(512, activation="relu", name="dense22")(t2)
  model2 = Model(input_tensor2, t2)
  t3 = Dense(512, activation="relu", name="dense3")(input_tensor3)
  t3 = Dense(512, activation="relu", name="dense33")(t3)
  model3 = Model(input_tensor3, t3)
  t4 = Dense(512, activation="relu", name="dense4")(input_tensor4)
  t4 = Dense(512, activation="relu", name="dense44")(t4)
  model4 = Model(input_tensor4, t4)
  
  input_tensor = Input(shape=(784,))
  t1 = model1(input_tensor)
  t2 = model2(input_tensor)
  t3 = model3(input_tensor)
  t4 = model4(input_tensor)
  output = Concatenate(axis=1)([t1, t2, t3, t4])
  #output2 = Dense(512, activation="relu")(output)
  output = Dense(num_classes)(output)
  output = Activation("softmax")(output)
  
  model = Model(input_tensor, output)

  opt = flexflow.keras.optimizers.SGD(learning_rate=0.01)
  model.compile(optimizer=opt)
  
  print(model.summary())

  model.fit(x_train, y_train, epochs=1)
  
def mlp_concat2():
  num_classes = 10

  (x_train, y_train), (x_test, y_test) = mnist.load_data()

  x_train = x_train.reshape(60000, 784)
  x_train = x_train.astype('float32')
  x_train /= 255
  y_train = y_train.astype('int32')
  y_train = np.reshape(y_train, (len(y_train), 1))
  #y_train = np.random.randint(1, 9, size=(len(y_train),1), dtype='int32')
  print("shape: ", x_train.shape)
  
  input_tensor1 = Input(shape=(256,))
  input_tensor2 = Input(shape=(256,))
  input_tensor3 = Input(shape=(256,))
  input_tensor4 = Input(shape=(256,))
  
  t1 = Dense(512, activation="relu", name="dense1")(input_tensor1)
  t1 = Dense(512, activation="relu", name="dense12")(t1)
  model1 = Model(input_tensor1, t1)
  t2 = Dense(512, activation="relu", name="dense2")(input_tensor2)
  t2 = Dense(512, activation="relu", name="dense22")(t2)
  model2 = Model(input_tensor2, t2)
  t3 = Dense(512, activation="relu", name="dense3")(input_tensor3)
  t3 = Dense(512, activation="relu", name="dense33")(t3)
  model3 = Model(input_tensor3, t3)
  t4 = Dense(512, activation="relu", name="dense4")(input_tensor4)
  t4 = Dense(512, activation="relu", name="dense44")(t4)
  model4 = Model(input_tensor4, t4)
  
  input_tensor = Input(shape=(784,))
  t00 = Input(shape=(512,))
  t01 = Input(shape=(512,))
  t1 = model1(input_tensor)
  t2 = model2(input_tensor)
  t3 = model3(input_tensor)
  t4 = model4(input_tensor)
  output = Concatenate(axis=1)([t00, t01, t1, t2, t3, t4])
  #output2 = Dense(512, activation="relu")(output)
  output = Dense(num_classes)(output)
  output = Activation("softmax")(output)
  
  model = Model([t00, t01, input_tensor], output)

  opt = flexflow.keras.optimizers.SGD(learning_rate=0.01)
  model.compile(optimizer=opt)
  
  print(model.summary())

  # model.fit(x_train, y_train, epochs=1)
   
def cnn():
  num_classes = 10

  img_rows, img_cols = 28, 28
  
  (x_train, y_train), (x_test, y_test) = mnist.load_data()
  x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
  
  x_train = x_train.astype('float32')
  x_train /= 255
  y_train = y_train.astype('int32')
  y_train = np.reshape(y_train, (len(y_train), 1))
  
  input_tensor = Input(shape=(1, 28, 28), dtype="float32")
  
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
  
  print(model.summary())
  
  flatten1 = model.get_layer(name='flat')
  t1 = flatten1.output_tensors[0]
  t2 = flatten1.input_tensors[0]
  print("t1: ", t1.to_layers, " ", t1.from_layer)
  print("t2: ", t2.to_layers, " ", t2.from_layer)
  
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
  
  input_tensor = Input(shape=(1, 28, 28), dtype="float32")
  
  t1 = Conv2D(filters=32, input_shape=(1,28,28), kernel_size=(3,3), strides=(1,1), padding=(1,1), activation="relu")(input_tensor)
  t2 = Conv2D(filters=32, input_shape=(1,28,28), kernel_size=(3,3), strides=(1,1), padding=(1,1), activation="relu")(input_tensor)
  output = concatenate([t1, t2])
  output = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding=(1,1), activation="relu")(output)
  output = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="valid")(output)
  output = Flatten()(output)
  output = Dense(128, activation="relu")(output)
  output = Dense(num_classes)(output)
  output = Activation("softmax")(output)

  model = Model(input_tensor, output)
  
  opt = flexflow.keras.optimizers.SGD(learning_rate=0.01)
  model.compile(optimizer=opt)
  print(model.summary())
  
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
  
  input_tensor1 = Input(shape=(3, 32, 32), dtype="float32")
  
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
  
  opt = flexflow.keras.optimizers.SGD(learning_rate=0.01)
  model.compile(optimizer=opt)
  print(model.summary())

  model.fit(x_train, y_train, epochs=1)


def cifar_cnn_sub(input_tensor, name_postfix):
  name = "conv2d_0_" + str(name_postfix)
  t1 = Conv2D(filters=32, input_shape=(3,32,32), kernel_size=(3,3), strides=(1,1), padding=(1,1), activation="relu", name=name)(input_tensor)
  name = "conv2d_1_" + str(name_postfix)
  ot1 = Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding=(1,1), activation="relu", name=name)(t1)
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
  
  input_tensor1 = Input(shape=(3, 32, 32), dtype="float32")
  input_tensor2 = Input(shape=(3, 32, 32), dtype="float32")

  ot1 = cifar_cnn_sub(input_tensor1, 1)
  ot2 = cifar_cnn_sub(input_tensor2, 2)
  ot3 = cifar_cnn_sub(input_tensor2, 3)
  output_tensor = Concatenate(axis=1)([ot1, ot2, ot3])
  output_tensor = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="valid")(output_tensor)
  o1 = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding=(1,1), activation="relu", name="conv2d_0_4")(output_tensor)
  o2 = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding=(1,1), activation="relu", name="conv2d_1_4")(output_tensor)
  output_tensor = Concatenate(axis=1)([o1, o2])
  output_tensor = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding=(1,1), activation="relu")(output_tensor)
  output_tensor = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="valid")(output_tensor)
  output_tensor = Flatten()(output_tensor)
  output_tensor = Dense(512, activation="relu")(output_tensor)
  output_tensor = Dense(num_classes)(output_tensor)
  output_tensor = Activation("softmax")(output_tensor)

  model = Model([input_tensor1, input_tensor2], output_tensor)
  
  opt = flexflow.keras.optimizers.SGD(learning_rate=0.01)
  model.compile(optimizer=opt)
  print(model.summary())

  model.fit([x_train, x_train], y_train, epochs=1)
  
def cifar_cnn_model_concat():
  num_classes = 10
  
  num_samples = 10000
  
  (x_train, y_train), (x_test, y_test) = cifar10.load_data(num_samples)
  
  x_train = x_train.astype('float32')
  x_train /= 255
  #x_train *= 0
  #y_train = np.random.randint(1, 9, size=(num_samples,1), dtype='int32')
  y_train = y_train.astype('int32')
  print("shape: ", x_train.shape)
  
  input_tensor1 = Input(shape=(3, 32, 32), dtype="float32")
  input_tensor2 = Input(shape=(3, 32, 32), dtype="float32")

  ot1 = cifar_cnn_sub(input_tensor1, 1)
  model1 = Model(input_tensor1, ot1)
  print(model1.summary())
  ot2 = cifar_cnn_sub(input_tensor2, 2)
  model2 = Model(input_tensor2, ot2)
  print(model2.summary())
  output_tensor = Concatenate(axis=1)([model1.output, model2.output])
  output_tensor = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="valid")(output_tensor)
  output_tensor = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding=(1,1), activation="relu", name="conv2d_0_4")(output_tensor)
  output_tensor = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding=(1,1), activation="relu")(output_tensor)
  output_tensor = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="valid")(output_tensor)
  output_tensor = Flatten()(output_tensor)
  output_tensor = Dense(512, activation="relu")(output_tensor)
  output_tensor = Dense(num_classes)(output_tensor)
  output_tensor = Activation("softmax")(output_tensor)

  model = Model([input_tensor1, input_tensor2], output_tensor)
  
  opt = flexflow.keras.optimizers.SGD(learning_rate=0.001)
  model.compile(optimizer=opt)
  print(model.summary())

  model.fit([x_train, x_train], y_train, epochs=1)
  
def cifar_cnn_model_concat_seq():
  num_classes = 10
  
  num_samples = 10000
  
  (x_train, y_train), (x_test, y_test) = cifar10.load_data(num_samples)
  
  x_train = x_train.astype('float32')
  x_train /= 255
  #x_train *= 0
  #y_train = np.random.randint(1, 9, size=(num_samples,1), dtype='int32')
  y_train = y_train.astype('int32')
  print("shape: ", x_train.shape)

  model1 = Sequential()
  model1.add(Conv2D(filters=32, input_shape=(3,32,32), kernel_size=(3,3), strides=(1,1), padding=(1,1), activation="relu", name="conv2d_0_0"))
  model1.add(Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding=(1,1), activation="relu", name="conv2d_1_0"))
  print(model1.summary())

  model2 = Sequential()
  model2.add(Conv2D(filters=32, input_shape=(3,32,32), kernel_size=(3,3), strides=(1,1), padding=(1,1), activation="relu", name="conv2d_0_1"))
  model2.add(Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding=(1,1), activation="relu", name="conv2d_1_1"))
  print(model2.summary())
  
  output_tensor = Concatenate(axis=1)([model1.output, model2.output])
  output_tensor = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="valid")(output_tensor)
  output_tensor = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding=(1,1), activation="relu", name="conv2d_0_4")(output_tensor)
  output_tensor = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding=(1,1), activation="relu")(output_tensor)
  output_tensor = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="valid")(output_tensor)
  output_tensor = Flatten()(output_tensor)
  output_tensor = Dense(512, activation="relu")(output_tensor)
  output_tensor = Dense(num_classes)(output_tensor)
  output_tensor = Activation("softmax")(output_tensor)

  model = Model([model1.input[0], model2.input[0]], output_tensor)
  
  opt = flexflow.keras.optimizers.SGD(learning_rate=0.001)
  model.compile(optimizer=opt)
  print(model.summary())

  model.fit([x_train, x_train], y_train, epochs=1)
  
def cifar_cnn_model_call():
  num_classes = 10
  
  num_samples = 10000
  
  (x_train, y_train), (x_test, y_test) = cifar10.load_data(num_samples)
  
  x_train = x_train.astype('float32')
  x_train /= 255
  #x_train *= 0
  #y_train = np.random.randint(1, 9, size=(num_samples,1), dtype='int32')
  y_train = y_train.astype('int32')
  print("shape: ", x_train.shape)
  
  input_tensor1 = Input(shape=(3, 32, 32), dtype="float32")
  output_tensor1 = Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding=(1,1), activation="relu")(input_tensor1)
  output_tensor1 = Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding=(1,1), activation="relu")(output_tensor1)
  output_tensor1 = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="valid")(output_tensor1)
  model1 = Model(input_tensor1, output_tensor1)
  
  input_tensor2 = Input(shape=(3, 32, 32), dtype="float32")
  output_tensor2 = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding=(1,1), activation="relu")(input_tensor2)
  output_tensor2 = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding=(1,1), activation="relu")(output_tensor2)
  output_tensor2 = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="valid")(output_tensor2)
  output_tensor2 = Flatten()(output_tensor2)
  output_tensor2 = Dense(512, activation="relu")(output_tensor2)
  output_tensor2 = Dense(num_classes)(output_tensor2)
  output_tensor2 = Activation("softmax")(output_tensor2)
  model2 = Model(input_tensor2, output_tensor2)
  
  input_tensor3 = Input(shape=(3, 32, 32), dtype="float32")
  output_tensor3 = model1(input_tensor3)
  output_tensor3 = model2(output_tensor3)
  model = Model(input_tensor3, output_tensor3)
  
  opt = flexflow.keras.optimizers.SGD(learning_rate=0.01)
  model.compile(optimizer=opt)
  print(model.summary())

  model.fit(x_train, y_train, epochs=1)
  
def cifar_alexnet():
  
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
  
  opt = flexflow.keras.optimizers.SGD(learning_rate=0.001)
  model.compile(optimizer=opt)
  print(model.summary())
  
  model.fit(full_input_np, full_label_np, epochs=1)

def mlp_net2net():
  num_classes = 10

  (x_train, y_train), (x_test, y_test) = mnist.load_data()

  x_train = x_train.reshape(60000, 784)
  x_train = x_train.astype('float32')
  x_train /= 255
  y_train = y_train.astype('int32')
  y_train = np.reshape(y_train, (len(y_train), 1))
  #y_train = np.random.randint(1, 9, size=(len(y_train),1), dtype='int32')
  print("shape: ", x_train.shape)
  
  #teacher
  
  input_tensor1 = Input(shape=(784,), dtype="float32")
  
  d1 = Dense(512, input_shape=(784,), activation="relu")
  d2 = Dense(512, activation="relu")
  d3 = Dense(num_classes)
  
  output = d1(input_tensor1)
  output = d2(output)
  output = d3(output)
  output = Activation("softmax")(output)
  
  teacher_model = Model(input_tensor1, output)

  opt = flexflow.keras.optimizers.SGD(learning_rate=0.01)
  teacher_model.compile(optimizer=opt)

  teacher_model.fit(x_train, y_train, epochs=1)
  
  d1_kernel, d1_bias = d1.get_weights(teacher_model.ffmodel)
  d2_kernel, d2_bias = d2.get_weights(teacher_model.ffmodel)
  d3_kernel, d3_bias = d3.get_weights(teacher_model.ffmodel)
  #d2_kernel_new = np.concatenate((d2_kernel, d2_kernel), axis=1)
  
  # student
  
  input_tensor2 = Input(shape=(784,), dtype="float32")
  
  sd1_1 = Dense(512, input_shape=(784,), activation="relu")
  #sd1_2 = Dense(512, input_shape=(784,), activation="relu")
  sd2 = Dense(512, activation="relu")
  sd3 = Dense(num_classes)
  
  output = sd1_1(input_tensor2)
  # t2 = sd1_2(input_tensor2)
  # output = Concatenate(axis=1)([t1, t2])
  output = sd2(output)
  output = sd3(output)
  output = Activation("softmax")(output)

  student_model = Model(input_tensor2, output)

  opt = flexflow.keras.optimizers.SGD(learning_rate=0.01)
  student_model.compile(optimizer=opt)
  
  sd1_1.set_weights(student_model.ffmodel, d1_kernel, d1_bias)
  # sd1_2.set_weights(student_model.ffmodel, d1_kernel, d1_bias)
  sd2.set_weights(student_model.ffmodel, d2_kernel, d2_bias)
  sd3.set_weights(student_model.ffmodel, d3_kernel, d3_bias)

  student_model.fit(x_train, y_train, epochs=1)

def cifar_cnn_net2net():
  num_classes = 10
  
  num_samples = 10000
  
  (x_train, y_train), (x_test, y_test) = cifar10.load_data(num_samples)
  
  x_train = x_train.astype('float32')
  x_train /= 255
  #x_train *= 0
  #y_train = np.random.randint(1, 9, size=(num_samples,1), dtype='int32')
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
  teacher_model.compile(optimizer=opt)
  print(teacher_model.summary())

  teacher_model.fit(x_train, y_train, epochs=1)

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
  student_model.compile(optimizer=opt)
  print(student_model.summary())

  sc1_1.set_weights(student_model.ffmodel, c1_kernel, c1_bias)
  sc1_2.set_weights(student_model.ffmodel, c1_kernel, c1_bias)
  sc2.set_weights(student_model.ffmodel, c2_kernel_new, c2_bias)
  sc3.set_weights(student_model.ffmodel, c3_kernel, c3_bias)
  sc4.set_weights(student_model.ffmodel, c4_kernel, c4_bias)
  sd1.set_weights(student_model.ffmodel, d1_kernel, d1_bias)
  sd2.set_weights(student_model.ffmodel, d2_kernel, d2_bias)

  student_model.fit(x_train, y_train, epochs=1)
    

def top_level_task():
  
  parser = argparse.ArgumentParser()
  parser.add_argument('--type', default=4)
  
  args, unknown = parser.parse_known_args()
  test_type = int(args.type)
  print(test_type)
  
  if (test_type == 1):
    cnn()
  elif (test_type == 2):
    cnn_concat()
  elif (test_type == 3):
    cifar_cnn()
  elif (test_type == 4):
    cifar_cnn_concat()
  elif (test_type == 5):
    cifar_alexnet()
  elif (test_type == 6):
    mlp()
  elif (test_type == 7):
    mlp_concat()
  elif (test_type == 8):
    mlp_net2net()
  elif (test_type == 9):
    cifar_cnn_net2net()
  elif (test_type == 10):
    cifar_cnn_model_concat()
  elif (test_type == 11):
    cifar_cnn_model_concat_seq()
  elif (test_type == 12):
    cifar_cnn_model_call()
  elif (test_type == 13):
    mlp_concat2()
    
  gc.collect()

if __name__ == "__main__":
  print("alexnet keras")
  top_level_task()