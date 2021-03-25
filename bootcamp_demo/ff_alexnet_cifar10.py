#./flexflow_python $FF_HOME/bootcamp_demo/ff_alexnet_cifar10.py -ll:py 1 -ll:gpu 1 -ll:fsize 2048 -ll:zsize 12192

from flexflow.core import *
from flexflow.keras.datasets import cifar10
from flexflow.torch.model import PyTorchModel
from PIL import Image

def top_level_task():
  ffconfig = FFConfig()
  ffconfig.parse_args()
  print("Python API batchSize(%d) workersPerNodes(%d) numNodes(%d)" %(ffconfig.get_batch_size(), ffconfig.get_workers_per_node(), ffconfig.get_num_nodes()))
  ffmodel = FFModel(ffconfig)

  dims_input = [ffconfig.get_batch_size(), 3, 229, 229]
  input_tensor = ffmodel.create_tensor(dims_input, DataType.DT_FLOAT)
  
  torch_model = PyTorchModel("alexnet.ff")  
  output_tensors = torch_model.apply(ffmodel, [input_tensor])

  ffoptimizer = SGDOptimizer(ffmodel, 0.01)
  ffmodel.set_sgd_optimizer(ffoptimizer)
  ffmodel.compile(loss_type=LossType.LOSS_SPARSE_CATEGORICAL_CROSSENTROPY, metrics=[MetricsType.METRICS_ACCURACY, MetricsType.METRICS_SPARSE_CATEGORICAL_CROSSENTROPY])
  label_tensor = ffmodel.get_label_tensor()

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
  dataloader_label = ffmodel.create_data_loader(label_tensor, full_label_np)

  num_samples = dataloader_input.num_samples

  ffmodel.init_layers()

  epochs = ffconfig.get_epochs()

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
