from flexflow.keras.models import Sequential
from flexflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation

import flexflow.core as ff

def top_level_task():
  model = Sequential()
  model.add(Conv2D(filters=64, input_shape=(229,229,3), kernel_size=(11,11), strides=(4,4), padding=(2,2)))
  model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding="valid"))
  model.add(Conv2D(filters=192, kernel_size=(5,5), strides=(1,1), padding=(2,2)))
  model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding="valid"))
  model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding=(1,1)))
  model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding=(1,1)))
  model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding=(1,1)))
  model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding="valid"))
  model.add(Flatten())
  model.add(Dense(4096, activation="relu"))
  model.add(Dense(4096, activation="relu"))
  model.add(Dense(1000))
  model.add(Activation("softmax"))
  
  # model.add_v2(Conv2D(filters=64, input_shape=(229,229,3), kernel_size=(11,11), strides=(4,4), padding=(2,2)))
  # model.add_v2(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding="valid"))
  # model.add_v2(Conv2D(filters=192, kernel_size=(5,5), strides=(1,1), padding=(2,2)))
  # model.add_v2(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding="valid"))
  # model.add_v2(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding=(1,1)))
  # model.add_v2(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding=(1,1)))
  # model.add_v2(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding=(1,1)))
  # model.add_v2(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding="valid"))
  # model.add_v2(Flatten())
  # model.add_v2(Dense(4096, activation="relu"))
  # model.add_v2(Dense(4096, activation="relu"))
  # model.add_v2(Dense(1000))
  # model.add_v2(Activation("softmax"))

  model.compile()

  dims = [model.ffconfig.get_batch_size(), 3, 229, 229]
  input = model.ffmodel.create_tensor_4d(dims, "", ff.DataType.DT_FLOAT);

  dims_label = [model.ffconfig.get_batch_size(), 1]
  label = model.ffmodel.create_tensor_2d(dims_label, "", ff.DataType.DT_INT32);
  dataloader = ff.DataLoader(model.ffmodel, input, label, 1)

  model.fit(input, label)

if __name__ == "__main__":
  print("alexnet keras")
  top_level_task()