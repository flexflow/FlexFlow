from flexflow.keras.models import Sequential
from flexflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation

import flexflow.core as ff

def top_level_task():
  model = Sequential()
  model.add(Dense(512, input_shape=(784,), activation="relu"))
  model.add(Dense(512, activation="relu"))
  model.add(Dense(10, activation="relu"))
  model.add(Activation("softmax"))

  model.compile()

  dims = [model.ffconfig.get_batch_size(), 784]
  input1 = model.ffmodel.create_tensor_2d(dims, "", ff.DataType.DT_FLOAT);
  
  dims_label = [model.ffconfig.get_batch_size(), 1]
  label = model.ffmodel.create_tensor_2d(dims_label, "", ff.DataType.DT_INT32);
  
  input1.inline_map(model.ffconfig)
  input1_array = input1.get_array(model.ffconfig, ff.DataType.DT_FLOAT)
  print(input1_array.shape)
  input1.inline_unmap(model.ffconfig)
  
  label.inline_map(model.ffconfig)
  label_array = label.get_array(model.ffconfig, ff.DataType.DT_INT32)
  print(label_array.shape)
  label.inline_unmap(model.ffconfig)

  model.fit(input1, label)

if __name__ == "__main__":
  print("alexnet keras")
  top_level_task()