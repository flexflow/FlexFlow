from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Input
import keras2onnx
import onnx

from keras import backend
backend.set_image_data_format('channels_first')

num_classes = 10

input_tensor = Input(shape=(784))
output = Dense(512, activation="relu")(input_tensor)
output = Dense(512, activation="relu")(output)
output = Dense(num_classes)(output)
output = Activation("softmax")(output)
model = Model(inputs=input_tensor, outputs=output)

# model = Sequential()
# model.add(Dense(512, input_shape=(64,784)))
# model.add(Activation('relu'))
# model.add(Dense(512, activation="relu"))
# model.add(Dense(num_classes))
# model.add(Activation("softmax"))

# layers = [Input(shape=(28, 28, 1,)),
#           Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding="valid", activation="relu"),
#           Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding="valid", activation="relu"),
#           MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="valid"),
#           Flatten(),
#           Dense(128, activation="relu"),
#           Dense(num_classes),
#           Activation("softmax")]
# model = Sequential(layers)

onnx_model = keras2onnx.convert_keras(model, "mlp")
onnx.save(onnx_model, "mnist_mlp_keras.onnx")

for node in onnx_model.graph.node:
  print(node)
#
# for input in onnx_model.graph.initializer:
#   print(input.name, input.dims, len(input.dims))
#   if '/bias' in input.name:
#     print(input.name, type(input))
#
# for input in onnx_model.graph.input:
#   print(input)
#
# for output in onnx_model.graph.output:
#   print(output, type(output))