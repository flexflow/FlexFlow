from flexflow.core import *
import numpy as np
from flexflow.keras.datasets import mnist

from accuracy import ModelAccuracy
import argparse


def top_level_task():
    ffconfig = FFConfig()
    print("Python API batchSize(%d) workersPerNodes(%d) numNodes(%d)" % (
        ffconfig.batch_size, ffconfig.workers_per_node, ffconfig.num_nodes))
    ffmodel = FFModel(ffconfig)

    dims_input = [ffconfig.batch_size, 784]
    input_tensor = ffmodel.create_tensor(dims_input, DataType.DT_FLOAT)

    num_samples = 60000

    kernel_init = UniformInitializer(12, -1, 1)
    t = ffmodel.dense(input_tensor, 512, ActiMode.AC_MODE_RELU,
                      kernel_initializer=kernel_init)
    t = ffmodel.dense(t, 512, ActiMode.AC_MODE_RELU)
    t = ffmodel.dense(t, 10)

    t = ffmodel.softmax(t)

    ffoptimizer = SGDOptimizer(ffmodel, 0.01)
    ffmodel.optimizer = ffoptimizer
    ffmodel.compile(loss_type=LossType.LOSS_SPARSE_CATEGORICAL_CROSSENTROPY, metrics=[
                    MetricsType.METRICS_ACCURACY, MetricsType.METRICS_SPARSE_CATEGORICAL_CROSSENTROPY])
    label_tensor = ffmodel.label_tensor

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    print(x_train.shape)
    x_train = x_train.reshape(60000, 784)
    x_train = x_train.astype('float32')
    x_train /= 255
    y_train = y_train.astype('int32')
    y_train = np.reshape(y_train, (len(y_train), 1))

    dataloader_input = ffmodel.create_data_loader(input_tensor, x_train)
    dataloader_label = ffmodel.create_data_loader(label_tensor, y_train)

    ffmodel.init_layers()

    dense1 = ffmodel.get_layer_by_id(0)
    print(dense1)
    print(dense1.get_weight_tensor())

if __name__ == "__main__":
    print("mnist mlp test weight")
    top_level_task()