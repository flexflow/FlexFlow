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

from flexflow.core import *
import numpy as np
from flexflow.keras.datasets import mnist

from accuracy import ModelAccuracy

def top_level_task():
  ffconfig = FFConfig()
  ffconfig.parse_args()
  print("Python API batchSize(%d) workersPerNodes(%d) numNodes(%d)" %(ffconfig.get_batch_size(), ffconfig.get_workers_per_node(), ffconfig.get_num_nodes()))
  ffmodel = FFModel(ffconfig)
  
  dims1 = [ffconfig.get_batch_size(), 1, 28, 28]
  input1 = ffmodel.create_tensor(dims1, "", DataType.DT_FLOAT);
  
  # dims_label = [ffconfig.get_batch_size(), 1]
  # label = ffmodel.create_tensor(dims_label, "", DataType.DT_INT32);
  
  num_samples = 60000
  
  t = ffmodel.conv2d(input1, 32, 3, 3, 1, 1, 1, 1, ActiMode.AC_MODE_RELU, True)
  t = ffmodel.conv2d(t, 64, 3, 3, 1, 1, 1, 1, ActiMode.AC_MODE_RELU, True)
  t = ffmodel.pool2d(t, 2, 2, 2, 2, 0, 0)
  t = ffmodel.flat(t);
  t = ffmodel.dense(t, 128, ActiMode.AC_MODE_RELU)
  t = ffmodel.dense(t, 10)
  t = ffmodel.softmax(t)

  ffoptimizer = SGDOptimizer(ffmodel, 0.01)
  ffmodel.set_sgd_optimizer(ffoptimizer)
  ffmodel.compile(loss_type=LossType.LOSS_SPARSE_CATEGORICAL_CROSSENTROPY, metrics=[MetricsType.METRICS_ACCURACY, MetricsType.METRICS_SPARSE_CATEGORICAL_CROSSENTROPY])
  label = ffmodel.get_label_tensor()

  img_rows, img_cols = 28, 28
  (x_train, y_train), (x_test, y_test) = mnist.load_data()
  x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
  x_train = x_train.astype('float32')
  x_train /= 255
  y_train = y_train.astype('int32')
  y_train = np.reshape(y_train, (len(y_train), 1))
  
  dims_full_input = [num_samples, 1, 28, 28]
  full_input = ffmodel.create_tensor(dims_full_input, "", DataType.DT_FLOAT)

  dims_full_label = [num_samples, 1]
  full_label = ffmodel.create_tensor(dims_full_label, "", DataType.DT_INT32)

  full_input.attach_numpy_array(ffconfig, x_train)
  full_label.attach_numpy_array(ffconfig, y_train)
  print(y_train)

  #dataloader = DataLoader2D(ffmodel, input1, label, full_input, full_label, num_samples)
  dataloader_input = SingleDataLoader(ffmodel, input1, full_input, num_samples, DataType.DT_FLOAT)
  dataloader_label = SingleDataLoader(ffmodel, label, full_label, num_samples, DataType.DT_INT32)

  full_input.detach_numpy_array(ffconfig)
  full_label.detach_numpy_array(ffconfig)

  ffmodel.init_layers()

  epochs = ffconfig.get_epochs()

  ts_start = ffconfig.get_current_time()
  for epoch in range(0,epochs):
    dataloader_input.reset()
    dataloader_label.reset()
    # dataloader.reset()
    ffmodel.reset_metrics()
    iterations = num_samples / ffconfig.get_batch_size()
    for iter in range(0, int(iterations)):
      dataloader_input.next_batch(ffmodel)
      dataloader_label.next_batch(ffmodel)
      #dataloader.next_batch(ffmodel)
      if (epoch > 0):
        ffconfig.begin_trace(111)
      ffmodel.forward()
      ffmodel.zero_gradients()
      ffmodel.backward()
      ffmodel.update()
      if (epoch > 0):
        ffconfig.end_trace(111)

  ts_end = ffconfig.get_current_time()
  run_time = 1e-6 * (ts_end - ts_start);
  print("epochs %d, ELAPSED TIME = %.4fs, THROUGHPUT = %.2f samples/s\n" %(epochs, run_time, num_samples * epochs / run_time));
  
  perf_metrics = ffmodel.get_perf_metrics()
  accuracy = perf_metrics.get_accuracy()
  if accuracy < ModelAccuracy.MNIST_CNN.value:
    assert 0, 'Check Accuracy'

  dense1 = ffmodel.get_layer_by_id(0)

  label.inline_map(ffconfig)
  label_array = label.get_array(ffconfig, DataType.DT_INT32)
  print(label_array.shape)
  print(label_array)
  label.inline_unmap(ffconfig)

  input1.inline_map(ffconfig)
  input1_array = input1.get_array(ffconfig, DataType.DT_FLOAT)
  print(input1_array.shape)
  print(input1_array[10, :, :, :])
  input1.inline_unmap(ffconfig)
  
  
if __name__ == "__main__":
  print("mnist mlp")
  top_level_task()
