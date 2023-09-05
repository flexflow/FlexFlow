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

from flexflow.core import *
from flexflow.keras.datasets import cifar10
from accuracy import ModelAccuracy

def next_batch(idx, x_train, input1, ffconfig, ffmodel):
  start = idx*ffconfig.batch_size
  x_train_batch = x_train[start:start+ffconfig.batch_size, :, :, :]
  print(x_train_batch.shape)

  # input1.inline_map(ffconfig)
  # input_array = input1.get_array(ffconfig, DataType.DT_FLOAT)
  # print(input_array.shape)
  # for i in range(0, ffconfig.batch_size):
  #   for j in range(0, 3):
  #     for k in range(0, 32):
  #       for l in range(0, 32):
  #         input_array[i][j][k][l] = x_train_batch[i][j][k][l]
  # input1.inline_unmap(ffconfig)

  input1.set_tensor(ffmodel, x_train_batch)

def next_batch_label(idx, x_train, input1, ffconfig, ffmodel):
  start = idx*ffconfig.batch_size
  x_train_batch = x_train[start:start+ffconfig.batch_size, :]
  print(x_train_batch.shape)

  # input1.inline_map(ffconfig)
  # input_array = input1.get_array(ffconfig, DataType.DT_INT32)
  # print(input_array.shape)
  # for i in range(0, ffconfig.batch_size):
  #   for j in range(0, 1):
  #     input_array[i][j] = x_train_batch[i][j]
  # input1.inline_unmap(ffconfig)
  input1.set_tensor(ffmodel, x_train_batch)

def top_level_task():
  ffconfig = FFConfig()
  alexnetconfig = NetConfig()
  print(alexnetconfig.dataset_path)
  print("Python API batchSize(%d) workersPerNodes(%d) numNodes(%d)" %(ffconfig.batch_size, ffconfig.workers_per_node, ffconfig.num_nodes))
  ffmodel = FFModel(ffconfig)

  dims_input = [ffconfig.batch_size, 3, 32, 32]
  input_tensor = ffmodel.create_tensor(dims_input, DataType.DT_FLOAT)

  num_samples = 10000

  (x_train, y_train), (x_test, y_test) = cifar10.load_data(num_samples)

  x_train = x_train.astype('float32')
  x_train /= 255

  full_input_array = x_train
  print(full_input_array.__array_interface__["strides"])

  y_train = y_train.astype('int32')

  full_label_array = y_train

  print(full_input_array.__array_interface__["strides"])
  print(full_input_array.shape, full_label_array.shape)
  print(full_label_array.__array_interface__["strides"])

  t = ffmodel.conv2d(input_tensor, 32, 3, 3, 1, 1, 1, 1, ActiMode.AC_MODE_RELU)
  t = ffmodel.conv2d(t, 32, 3, 3, 1, 1, 1, 1, ActiMode.AC_MODE_RELU)
  t = ffmodel.pool2d(t, 2, 2, 2, 2, 0, 0,)
  t = ffmodel.conv2d(t, 64, 3, 3, 1, 1, 1, 1, ActiMode.AC_MODE_RELU)
  t = ffmodel.conv2d(t, 64, 3, 3, 1, 1, 1, 1, ActiMode.AC_MODE_RELU)
  t = ffmodel.pool2d(t, 2, 2, 2, 2, 0, 0)
  t = ffmodel.flat(t);
  t = ffmodel.dense(t, 512, ActiMode.AC_MODE_RELU)
  t = ffmodel.dense(t, 10)
  t = ffmodel.softmax(t)

  ffoptimizer = SGDOptimizer(ffmodel, 0.01)
  ffmodel.optimizer = ffoptimizer
  ffmodel.compile(loss_type=LossType.LOSS_SPARSE_CATEGORICAL_CROSSENTROPY, metrics=[MetricsType.METRICS_ACCURACY, MetricsType.METRICS_SPARSE_CATEGORICAL_CROSSENTROPY])
  label_tensor = ffmodel.label_tensor

  next_batch(0, x_train, input_tensor, ffconfig, ffmodel)
  next_batch_label(0, y_train, label_tensor, ffconfig, ffmodel)

  ffmodel.init_layers()

  epochs = ffconfig.epochs

  ts_start = ffconfig.get_current_time()
  for epoch in range(0,epochs):
    ffmodel.reset_metrics()
    iterations = int(num_samples / ffconfig.batch_size)
    print(iterations, num_samples)
    ct = 0
    for iter in range(0, int(iterations)):
      #ffconfig.begin_trace(111)
      next_batch(ct, x_train, input_tensor, ffconfig, ffmodel)
      next_batch_label(ct, y_train, label_tensor, ffconfig, ffmodel)
      ct += 1
      ffmodel.forward()
      ffmodel.zero_gradients()
      ffmodel.backward()
      ffmodel.update()
      #ffconfig.end_trace(111)

  ts_end = ffconfig.get_current_time()
  run_time = 1e-6 * (ts_end - ts_start);
  print("epochs %d, ELAPSED TIME = %.4fs, THROUGHPUT = %.2f samples/s\n" %(epochs, run_time, num_samples * epochs / run_time));

  perf_metrics = ffmodel.get_perf_metrics()
  accuracy = perf_metrics.get_accuracy()
  if accuracy < 0.3:
    assert 0, 'Check Accuracy'

  conv_2d1 = ffmodel.get_layer_by_id(0)
  cbias_tensor = conv_2d1.get_input_tensor()
  #cbias_tensor = conv_2d1.get_output_tensor()
  cbias_tensor.inline_map(ffmodel, ffconfig)
  cbias = cbias_tensor.get_flat_array(ffmodel, ffconfig)
  print(cbias.shape)
  print(cbias)
  cbias_tensor.inline_unmap(ffmodel, ffconfig)

  label_tensor.inline_map(ffmodel, ffconfig)
  label_array = label_tensor.get_flat_array(ffmodel, ffconfig)
  print(label_array.shape)
  # print(cbias)
  print(label_array)
  label_tensor.inline_unmap(ffmodel, ffconfig)


if __name__ == "__main__":
  print("cifar10 cnn attach")
  configs = get_configs()
  init_flexflow_runtime(configs)
  top_level_task()
