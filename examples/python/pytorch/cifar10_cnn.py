from flexflow.core import *
from flexflow.keras.datasets import cifar10
from flexflow.torch.model import file_to_ff

#from accuracy import ModelAccuracy

def top_level_task():
  ffconfig = FFConfig()
  print("Python API batchSize(%d) workersPerNodes(%d) numNodes(%d)" %(ffconfig.batch_size, ffconfig.workers_per_node, ffconfig.num_nodes))
  ffmodel = FFModel(ffconfig)

  dims_input = [ffconfig.batch_size, 3, 32, 32]
  input_tensor = ffmodel.create_tensor(dims_input, DataType.DT_FLOAT)
  output_tensors = file_to_ff("cnn.ff", ffmodel, [input_tensor, input_tensor])

  t = ffmodel.softmax(output_tensors[0])

  ffoptimizer = SGDOptimizer(ffmodel, 0.01)
  ffmodel.optimizer = ffoptimizer
  ffmodel.compile(loss_type=LossType.LOSS_SPARSE_CATEGORICAL_CROSSENTROPY, metrics=[MetricsType.METRICS_ACCURACY, MetricsType.METRICS_SPARSE_CATEGORICAL_CROSSENTROPY])
  label_tensor = ffmodel.label_tensor

  num_samples = 10000

  (x_train, y_train), (x_test, y_test) = cifar10.load_data(num_samples)

  x_train = x_train.astype('float32')
  x_train /= 255
  full_input_array = x_train

  y_train = y_train.astype('int32')
  full_label_array = y_train

  dataloader_input = ffmodel.create_data_loader(input_tensor, full_input_array)
  dataloader_label = ffmodel.create_data_loader(label_tensor, full_label_array)

  num_samples = dataloader_input.num_samples

  ffmodel.init_layers()

  layers = ffmodel.get_layers()
  for layer in layers:
    print(layers[layer].name)

  layer = ffmodel.get_layer_by_name("relu_1")
  print(layer)

  epochs = ffconfig.epochs

  ts_start = ffconfig.get_current_time()
  
  ffmodel.fit(x=dataloader_input, y=dataloader_label, epochs=epochs)

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

