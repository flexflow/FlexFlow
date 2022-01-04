from flexflow.core import *
from flexflow.keras.datasets import cifar10
from flexflow.torch.model import PyTorchModel

from PIL import Image
import numpy as np

def top_level_task():
  ffconfig = FFConfig()
  alexnetconfig = NetConfig()
  print(alexnetconfig.dataset_path)
  print("Python API batchSize(%d) workersPerNodes(%d) numNodes(%d)" %(ffconfig.batch_size, ffconfig.workers_per_node, ffconfig.num_nodes))
  ffmodel = FFModel(ffconfig)
  
  dims_input = [ffconfig.batch_size, 3, 229, 229]
  input_tensor = ffmodel.create_tensor(dims_input, DataType.DT_FLOAT)

  output_tensors = PyTorchModel.file_to_ff("squeezenet.ff", ffmodel, [input_tensor])

  ffoptimizer = SGDOptimizer(ffmodel, 0.01)
  ffmodel.optimizer = ffoptimizer
  ffmodel.compile(loss_type=LossType.LOSS_SPARSE_CATEGORICAL_CROSSENTROPY, metrics=[MetricsType.METRICS_ACCURACY, MetricsType.METRICS_SPARSE_CATEGORICAL_CROSSENTROPY])
  label = ffmodel.label_tensor
  
  num_samples = 10000
  
  (x_train, y_train), (x_test, y_test) = cifar10.load_data(num_samples)

  full_input_np = np.zeros((num_samples, 3, 229, 229), dtype=np.float32)
  
  for i in range(0, num_samples):
    image = x_train[i, :, :, :]
    image = image.transpose(1, 2, 0)
    pil_image = Image.fromarray(image)
    pil_image = pil_image.resize((229,229), Image.NEAREST)
    image = np.array(pil_image, dtype=np.float32)
    image = image.transpose(2, 0, 1)
    full_input_np[i, :, :, :] = image
  
  full_input_np /= 255
  
  y_train = y_train.astype('int32')
  full_label_np = y_train
  
  dataloader_input = ffmodel.create_data_loader(input_tensor, full_input_np)
  dataloader_label = ffmodel.create_data_loader(label, full_label_np)
  
  num_samples = dataloader_input.num_samples
  assert dataloader_input.num_samples == dataloader_label.num_samples

  ffmodel.init_layers()

  epochs = ffconfig.epochs

  ts_start = ffconfig.get_current_time()
  
  ffmodel.fit(x=dataloader_input, y=dataloader_label, epochs=epochs)

  ts_end = ffconfig.get_current_time()
  run_time = 1e-6 * (ts_end - ts_start);
  print("epochs %d, ELAPSED TIME = %.4fs, THROUGHPUT = %.2f samples/s\n" %(epochs, run_time, num_samples * epochs / run_time));
  # perf_metrics = ffmodel.get_perf_metrics()
  # accuracy = perf_metrics.get_accuracy()
  # if accuracy < ModelAccuracy.CIFAR10_ALEXNET.value:
  #   assert 0, 'Check Accuracy'

if __name__ == "__main__":
  print("alexnet onnx")
  top_level_task()
