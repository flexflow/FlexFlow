from flexflow.core import *
from flexflow.keras.datasets import cifar10

#from accuracy import ModelAccuracy

def top_level_task():
  ffconfig = FFConfig()
  ffconfig.parse_args()
  print("Python API batchSize(%d) workersPerNodes(%d) numNodes(%d)" %(ffconfig.get_batch_size(), ffconfig.get_workers_per_node(), ffconfig.get_num_nodes()))
  ffmodel = FFModel(ffconfig)

  dims_input = [ffconfig.get_batch_size(), 3, 32, 32]
  input_tensor = ffmodel.create_tensor(dims_input, DataType.DT_FLOAT)

  output_tensors = ffmodel.construct_model_from_file([input_tensor, input_tensor], "cnn.ff")

  t = ffmodel.softmax(output_tensors[0])

  ffoptimizer = SGDOptimizer(ffmodel, 0.01)
  ffmodel.set_sgd_optimizer(ffoptimizer)
  ffmodel.compile(loss_type=LossType.LOSS_SPARSE_CATEGORICAL_CROSSENTROPY, metrics=[MetricsType.METRICS_ACCURACY, MetricsType.METRICS_SPARSE_CATEGORICAL_CROSSENTROPY])
  label = ffmodel.get_label_tensor()

  num_samples = 10000

  (x_train, y_train), (x_test, y_test) = cifar10.load_data(num_samples)

  x_train = x_train.astype('float32')
  x_train /= 255
  full_input_array = x_train

  y_train = y_train.astype('int32')
  full_label_array = y_train

  dims_full_input = [num_samples, 3, 32, 32]
  full_input = ffmodel.create_tensor(dims_full_input, DataType.DT_FLOAT)

  dims_full_label = [num_samples, 1]
  full_label = ffmodel.create_tensor(dims_full_label, DataType.DT_INT32)

  full_input.attach_numpy_array(ffconfig, full_input_array)
  full_label.attach_numpy_array(ffconfig, full_label_array)

  dataloader_input = SingleDataLoader(ffmodel, input_tensor, full_input, num_samples, DataType.DT_FLOAT)
  dataloader_label = SingleDataLoader(ffmodel, label, full_label, num_samples, DataType.DT_INT32)

  full_input.detach_numpy_array(ffconfig)
  full_label.detach_numpy_array(ffconfig)

  num_samples = dataloader_input.get_num_samples()

  ffmodel.init_layers()

  layers = ffmodel.get_layers()
  for layer in layers:
    print(layers[layer].name)

  layer = ffmodel.get_layer_by_name("relu_1")
  print(layer)

  epochs = ffconfig.get_epochs()

  ts_start = ffconfig.get_current_time()
  for epoch in range(0,epochs):
    dataloader_input.reset()
    dataloader_label.reset()
    ffmodel.reset_metrics()
    iterations = int(num_samples / ffconfig.get_batch_size())
    print(iterations, num_samples)

    for iter in range(0, int(iterations)):
      dataloader_input.next_batch(ffmodel)
      dataloader_label.next_batch(ffmodel)
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

  # perf_metrics = ffmodel.get_perf_metrics()
  # accuracy = perf_metrics.get_accuracy()
  # if accuracy < ModelAccuracy.CIFAR10_CNN.value:
  #   assert 0, 'Check Accuracy'


if __name__ == "__main__":
  print("cifar10 cnn")
  top_level_task()
