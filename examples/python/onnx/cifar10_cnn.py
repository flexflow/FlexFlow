from flexflow.core import *
from flexflow.keras.datasets import cifar10
from flexflow.onnx.model import ONNXModel, ONNXModelKeras
import argparse

from accuracy import ModelAccuracy

def top_level_task(test_type=1):
  ffconfig = FFConfig()
  alexnetconfig = NetConfig()
  print(alexnetconfig.dataset_path)
  print("Python API batchSize(%d) workersPerNodes(%d) numNodes(%d)" %(ffconfig.batch_size, ffconfig.workers_per_node, ffconfig.num_nodes))
  ffmodel = FFModel(ffconfig)
  
  dims_input = [ffconfig.batch_size, 3, 32, 32]
  input = ffmodel.create_tensor(dims_input, DataType.DT_FLOAT)

  if test_type == 1:
    onnx_model = ONNXModel("cifar10_cnn_pt.onnx")
    t = onnx_model.apply(ffmodel, {"input.1": input})
  else:
    onnx_model = ONNXModelKeras("cifar10_cnn_keras.onnx", ffconfig, ffmodel)
    t = onnx_model.apply(ffmodel, {"input_1": input})

  ffoptimizer = SGDOptimizer(ffmodel, 0.01)
  ffmodel.optimizer = ffoptimizer
  ffmodel.compile(loss_type=LossType.LOSS_SPARSE_CATEGORICAL_CROSSENTROPY, metrics=[MetricsType.METRICS_ACCURACY, MetricsType.METRICS_SPARSE_CATEGORICAL_CROSSENTROPY])
  label = ffmodel.label_tensor
  
  num_samples = 10000
  
  (x_train, y_train), (x_test, y_test) = cifar10.load_data(num_samples)
  
  x_train = x_train.astype('float32')
  x_train /= 255
  full_input_array = x_train
  print(full_input_array.__array_interface__["strides"])
  
  y_train = y_train.astype('int32')
  full_label_array = y_train
  
  dataloader_input = ffmodel.create_data_loader(input, full_input_array)
  dataloader_label = ffmodel.create_data_loader(label, full_label_array)
  
  num_samples = dataloader_input.num_samples

  ffmodel.init_layers()

  epochs = ffconfig.epochs

  ts_start = ffconfig.get_current_time()
  
  ffmodel.fit(x=dataloader_input, y=dataloader_label, epochs=epochs)

  ts_end = ffconfig.get_current_time()
  run_time = 1e-6 * (ts_end - ts_start);
  print("epochs %d, ELAPSED TIME = %.4fs, THROUGHPUT = %.2f samples/s\n" %(epochs, run_time, num_samples * epochs / run_time));
  
  perf_metrics = ffmodel.get_perf_metrics()
  accuracy = perf_metrics.get_accuracy()
  if accuracy < ModelAccuracy.CIFAR10_CNN.value:
    assert 0, 'Check Accuracy'

if __name__ == "__main__":
  print("cifar10 cnn onnx")
  parser = argparse.ArgumentParser()
  parser.add_argument('--test_type', type=int, choices=[0, 1], help="Whether to test using Keras (test_type 0) or PyTorch (test_type 1) ")
  args, unknown = parser.parse_known_args()
  test_type = args.test_type
  top_level_task(test_type)
