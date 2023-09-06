from flexflow.core import *

import numpy as np

def top_level_task():
  ffconfig = FFConfig()
  print("Python API batchSize(%d) workersPerNodes(%d) numNodes(%d)" %(ffconfig.batch_size, ffconfig.workers_per_node, ffconfig.num_nodes))
  ffmodel = FFModel(ffconfig)

  dims1 = [ffconfig.batch_size, 3, 229, 229]
  input1 = ffmodel.create_tensor(dims1, DataType.DT_FLOAT);

  dims2 = [ffconfig.batch_size, 16]
  input2 = ffmodel.create_tensor(dims2, DataType.DT_FLOAT);

  dims_label = [ffconfig.batch_size, 1]
  label = ffmodel.create_tensor(dims_label, DataType.DT_INT32);

  t1 = ffmodel.conv2d(input1, 64, 11, 11, 4, 4, 2, 2)
  t2 = ffmodel.dense(input2, 8, ActiMode.AC_MODE_RELU)
  #t3 = ffmodel.dense("dense1", t2, 128, ActiMode.AC_MODE_RELU)
  ffoptimizer = SGDOptimizer(ffmodel, 0.01)
  ffmodel.compile(optimizer=ffoptimizer, loss_type=LossType.LOSS_SPARSE_CATEGORICAL_CROSSENTROPY, metrics=[MetricsType.METRICS_ACCURACY, MetricsType.METRICS_SPARSE_CATEGORICAL_CROSSENTROPY])
  label = ffmodel.label_tensor

  # Data Loader
  # alexnetconfig = NetConfig()
  # dataloader = DataLoader4D(ffmodel, input1, label, ffnetconfig=alexnetconfig)

  # ffmodel.init_layers()

  label.inline_map(ffmodel, ffconfig)
  label_array = label.get_array(ffmodel, ffconfig)
  label_array *= 0
  label_array += 1
  print(label_array.shape)
  print(label_array)
  label.inline_unmap(ffmodel, ffconfig)

  #weight of conv2d
  # t3 = ffmodel.get_tensor_by_id(0)
  #
  # np_array = np.zeros((64, 3, 11, 11), dtype=np.float32)
  # np_array += 1.2
  # t3.set_weights(ffmodel, np_array)
  #
  # t3.inline_map(ffconfig)
  # t3_array = t3.get_array(ffconfig)
  # print(t3_array.shape)
  # print(t3_array)
  # t3.inline_unmap(ffconfig)


  ###3
  conv_2d1 = ffmodel.get_layer_by_id(0)

  cbias_tensor = conv_2d1.get_bias_tensor()

  np_array = np.zeros((64), dtype=np.float32)
  np_array += 22.222
  cbias_tensor.set_weights(ffmodel, np_array)

  # cbias_tensor.inline_map(ffconfig)
  # cbias = cbias_tensor.get_array(ffconfig)
  # print(cbias)
  # cbias *= 0.0
  # cbias += 1.1
  # print(cbias.shape)
  # print(cbias)
  # cbias_tensor.inline_unmap(ffconfig)
  #
  # cweight_tensor = conv_2d1.get_weight_tensor()
  # cweight_tensor.inline_map(ffconfig)
  # cweight = cweight_tensor.get_array(ffconfig)
  # #cweight += 1.2
  # ct = 0.0
  # for i in range(cweight.shape[0]):
  #   for j in range(cweight.shape[1]):
  #     for k in range(cweight.shape[2]):
  #       for l in range(cweight.shape[3]):
  #         cweight[i][j][k][l] += ct
  #         ct += 1.0
  # print(cweight.shape)
  # # print(cweight.strides)
  # print(cweight)
  # cweight_tensor.inline_unmap(ffconfig)
  #
  # np_array = cweight_tensor.get_weights(ffmodel)
  # print(np_array)

  dense1 = ffmodel.get_layer_by_id(1)

  dbias_tensor = dense1.get_bias_tensor()
  dbias_tensor.inline_map(ffmodel, ffconfig)
  dbias = dbias_tensor.get_array(ffmodel, ffconfig)
  dbias *= 0.0
  dbias += 2.1
  print(dbias.shape)
  print(dbias)
  dbias_tensor.inline_unmap(ffmodel, ffconfig)

  np_array = dbias_tensor.get_weights(ffmodel)
  print(np_array)

  dweight_tensor = dense1.get_weight_tensor()
  dweight_tensor.inline_map(ffmodel, ffconfig)
  dweight = dweight_tensor.get_array(ffmodel, ffconfig)
  #dweight *= 0.0
  #dweight += 2.2
  ct = 0.0
  # for i in range(dweight.shape[0]):
  #   for j in range(dweight.shape[1]):
  #     dweight[i][j] = ct
  #     ct += 1.0
  # print(dweight.shape)
  # print(dweight.strides)
  # print(dweight)
  dweight_tensor.inline_unmap(ffmodel, ffconfig)

  # ffmodel.print_layers(0)


if __name__ == "__main__":
  print("alexnet")
  configs = get_configs()
  init_flexflow_runtime(configs)
  top_level_task()
