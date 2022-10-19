from flexflow.core import *
import numpy as np
from flexflow.keras.datasets import mnist
from flexflow.onnx.model import ONNXModel, ONNXModelKeras
import argparse

from accuracy import ModelAccuracy

def top_level_task(test_type=1):
  ffconfig = FFConfig()
  print("Python API batchSize(%d) workersPerNodes(%d) numNodes(%d)" %(ffconfig.batch_size, ffconfig.workers_per_node, ffconfig.num_nodes))
  ffmodel = FFModel(ffconfig)
  
  dims1 = [ffconfig.batch_size, 784]
  input1 = ffmodel.create_tensor(dims1, DataType.DT_FLOAT);
  
  num_samples = 60000
  
  if test_type == 1:
    onnx_model = ONNXModel("mnist_mlp_pt.onnx")
    t = onnx_model.apply(ffmodel, {"input.1": input1})
  else:
    onnx_model = ONNXModelKeras("mnist_mlp_keras.onnx", ffconfig, ffmodel)
    t = onnx_model.apply(ffmodel, {"input_1": input1})

  ffoptimizer = SGDOptimizer(ffmodel, 0.01)
  ffmodel.optimizer = ffoptimizer
  ffmodel.compile(loss_type=LossType.LOSS_SPARSE_CATEGORICAL_CROSSENTROPY, metrics=[MetricsType.METRICS_ACCURACY, MetricsType.METRICS_SPARSE_CATEGORICAL_CROSSENTROPY])
  label = ffmodel.label_tensor

  (x_train, y_train), (x_test, y_test) = mnist.load_data()

  x_train = x_train.reshape(60000, 784)
  x_train = x_train.astype('float32')
  x_train /= 255
  y_train = y_train.astype('int32')
  y_train = np.reshape(y_train, (len(y_train), 1))

  dataloader_input = ffmodel.create_data_loader(input1, x_train)
  dataloader_label = ffmodel.create_data_loader(label, y_train)

  ffmodel.init_layers()

  epochs = ffconfig.epochs

  ts_start = ffconfig.get_current_time()

  ffmodel.fit(x=dataloader_input, y=dataloader_label, epochs=epochs)

  ts_end = ffconfig.get_current_time()
  run_time = 1e-6 * (ts_end - ts_start);
  print("epochs %d, ELAPSED TIME = %.4fs, THROUGHPUT = %.2f samples/s\n" %(epochs, run_time, num_samples * epochs / run_time));

  perf_metrics = ffmodel.get_perf_metrics()
  accuracy = perf_metrics.get_accuracy()
  if accuracy < ModelAccuracy.MNIST_MLP.value:
    assert 0, 'Check Accuracy'
  
if __name__ == "__main__":
  print("mnist mlp onnx")
  parser = argparse.ArgumentParser()
  parser.add_argument('--test_type', type=int, choices=[0, 1], help="Whether to test using Keras (test_type 0) or PyTorch (test_type 1) ")
  args, unknown = parser.parse_known_args()
  test_type = args.test_type
  top_level_task(test_type)
