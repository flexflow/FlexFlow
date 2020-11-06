from flexflow.core import *
from flexflow.keras.datasets import cifar10

from PIL import Image

def BottleneckBlock(ff, input, out_channels, stride):
  t = ff.conv2d(input, out_channels, 1, 1, 1, 1, 0, 0, ActiMode.AC_MODE_NONE)
  t = ff.batch_norm(t)
  t = ff.conv2d(t, out_channels, 3, 3, stride, stride, 1, 1, ActiMode.AC_MODE_NONE)
  t = ff.batch_norm(t)
  t = ff.conv2d(t, 4*out_channels, 1, 1, 1, 1, 0, 0)
  t = ff.batch_norm(t, False)
  if ((stride > 1) or (input.dims[1] != out_channels * 4)):
    print("input.adim = %d out_channels*4 = %d" %(input.dims[1], out_channels*4))
    input = ff.conv2d(input, 4*out_channels, 1, 1, stride, stride, 0, 0, ActiMode.AC_MODE_NONE)
    input = ff.batch_norm(input, False);
  t = ff.add(input, t)
  return ff.relu(t)

def top_level_task():
  ffconfig = FFConfig()
  alexnetconfig = NetConfig()
  print(alexnetconfig.dataset_path)
  ffconfig.parse_args()
  print("Python API batchSize(%d) workersPerNodes(%d) numNodes(%d)" %(ffconfig.get_batch_size(), ffconfig.get_workers_per_node(), ffconfig.get_num_nodes()))
  ffmodel = FFModel(ffconfig)

  dims_input = [ffconfig.get_batch_size(), 3, 229, 229]
  #print(dims)
  input = ffmodel.create_tensor(dims_input, DataType.DT_FLOAT)

  t = ffmodel.conv2d(input, 64, 7, 7, 2, 2, 3, 3)
  t = ffmodel.batch_norm(t);
  t = ffmodel.pool2d(t, 3, 3, 2, 2, 1, 1)
  for i in range(0, 3):
    t = BottleneckBlock(ffmodel, t, 64, 1)
  for i in range(0, 4):
    if (i == 0):
      stride = 2
    else:
      stride = 1
    t = BottleneckBlock(ffmodel, t, 128, stride)
  for i in range(0, 6):
    if (i == 0):
      stride = 2
    else:
      stride = 1
    t = BottleneckBlock(ffmodel, t, 256, stride)
  for i in range(0, 3):
    if (i == 0):
      stride = 2
    else:
      stride = 1
    t = BottleneckBlock(ffmodel, t, 512, stride);
  t = ffmodel.pool2d(t, 7, 7, 1, 1, 0, 0, PoolType.POOL_AVG)
  t = ffmodel.flat(t);
  t = ffmodel.dense(t, 10)
  t = ffmodel.softmax(t)

  ffoptimizer = SGDOptimizer(ffmodel, 0.001)
  ffmodel.set_sgd_optimizer(ffoptimizer)
  ffmodel.compile(loss_type=LossType.LOSS_SPARSE_CATEGORICAL_CROSSENTROPY, metrics=[MetricsType.METRICS_ACCURACY, MetricsType.METRICS_SPARSE_CATEGORICAL_CROSSENTROPY])
  label = ffmodel.get_label_tensor()

  # load data
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
  print(full_input_np.shape)
  print(full_input_np.__array_interface__["strides"])
  print(full_input_np[0,:, :, :])

  y_train = y_train.astype('int32')
  full_label_np = y_train

  dims_full_input = [num_samples, 3, 229, 229]
  full_input = ffmodel.create_tensor(dims_full_input, "", DataType.DT_FLOAT)

  dims_full_label = [num_samples, 1]
  full_label = ffmodel.create_tensor(dims_full_label, "", DataType.DT_INT32)

  full_input.attach_numpy_array(ffconfig, full_input_np)
  full_label.attach_numpy_array(ffconfig, full_label_np)

  dataloader_input = SingleDataLoader(ffmodel, input, full_input, num_samples, DataType.DT_FLOAT)
  dataloader_label = SingleDataLoader(ffmodel, label, full_label, num_samples, DataType.DT_INT32)

  full_input.detach_numpy_array(ffconfig)
  full_label.detach_numpy_array(ffconfig)

  num_samples = dataloader_input.get_num_samples()
  assert dataloader_input.get_num_samples() == dataloader_label.get_num_samples()

  ffmodel.init_layers()

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

if __name__ == "__main__":
  print("resnet")
  top_level_task()
