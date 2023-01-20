from flexflow.core import *
import numpy as np

def top_level_task():
  ffconfig = FFConfig()
  bs = ffconfig.batch_size
  ffmodel = FFModel(ffconfig)
  neighbors = [[[0], [5], [3], [3], [7], [9]]]
  neighbors = np.array(neighbors).repeat(bs, 0).repeat(5, 2)
  print(neighbors.shape)
  x = np.array([[[0.01 for i in range(5)] for j in range(16)] for k in range(bs)], np.single)
  print(x)
  input = ffmodel.create_tensor([bs, 16, 5], DataType.DT_FLOAT)
  index = ffmodel.create_tensor([bs, 6, 5], DataType.DT_INT32)
  x0 = ffmodel.dense(input, 5, ActiMode.AC_MODE_NONE, False)
  x1 = ffmodel.gather(x0, index, 1)

  ffoptimizer = SGDOptimizer(ffmodel, 0.01)
  ffmodel.optimizer = ffoptimizer
  ffmodel.compile(loss_type=LossType.LOSS_MEAN_SQUARED_ERROR_AVG_REDUCE, metrics=[MetricsType.METRICS_MEAN_SQUARED_ERROR])
  ffmodel.init_layers()
  input.attach_numpy_array(ffmodel, ffconfig, x)
  index.attach_numpy_array(ffmodel, ffconfig, neighbors)
  label_tensor = ffmodel.label_tensor
  y = np.random.rand(bs, 6, 5).astype('float32')
  label_tensor.attach_numpy_array(ffmodel, ffconfig, y)

  for _ in range(100):
    ffmodel.forward()
    ffmodel.backward()
    ffmodel.update()

if __name__ == "__main__":
  print("Demo Gather")
  top_level_task()
