from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Input
import keras2onnx
import onnx

from keras import backend
backend.set_image_data_format('channels_first')

num_classes = 10

input_tensor1 = Input(shape=(3, 32, 32))

output_tensor = Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding="valid", activation="relu")(input_tensor1)
output_tensor = Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding="valid", activation="relu")(output_tensor)
output_tensor = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="valid")(output_tensor)
output_tensor = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding="valid", activation="relu")(output_tensor)
output_tensor = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding="valid", activation="relu")(output_tensor)
output_tensor = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="valid")(output_tensor)
output_tensor = Flatten()(output_tensor)
output_tensor = Dense(512, activation="relu")(output_tensor)
output_tensor = Dense(num_classes)(output_tensor)
output_tensor = Activation("softmax")(output_tensor)

model = Model(input_tensor1, output_tensor)

print(model.summary())
print(model.get_layer(index=1).output.name)
print(model.get_layer(index=1).input.name)

onnx_model = keras2onnx.convert_keras(model, "mlp")
onnx.save(onnx_model, "cifar10_cnn_keras.onnx")

for node in onnx_model.graph.node:
  print(node)
#
# for input in onnx_model.graph.initializer:
#   print(input.name, input.dims, len(input.dims))
#
# for input in onnx_model.graph.input:
#   print(input)
#
# for output in onnx_model.graph.output:
#   print(output, type(output))