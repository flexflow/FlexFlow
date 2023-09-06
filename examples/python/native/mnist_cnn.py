# Copyright 2023 CMU, Facebook, LANL, MIT, NVIDIA, and Stanford (alphabetical)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from flexflow.core import *
import numpy as np
from flexflow.keras.datasets import mnist

from accuracy import ModelAccuracy
import argparse, json


def top_level_task():
    ffconfig = FFConfig()
    print("Python API batchSize(%d) workersPerNodes(%d) numNodes(%d)" % (
        ffconfig.batch_size, ffconfig.workers_per_node, ffconfig.num_nodes))
    ffmodel = FFModel(ffconfig)

    dims_input = [ffconfig.batch_size, 1, 28, 28]
    input_tensor = ffmodel.create_tensor(dims_input, DataType.DT_FLOAT)

    num_samples = 60000

    t = ffmodel.conv2d(input_tensor, 32, 3, 3, 1, 1, 1,
                       1, ActiMode.AC_MODE_RELU, True)
    t = ffmodel.conv2d(t, 64, 3, 3, 1, 1, 1, 1, ActiMode.AC_MODE_RELU, True)
    t = ffmodel.pool2d(t, 2, 2, 2, 2, 0, 0)
    t = ffmodel.flat(t)
    t = ffmodel.dense(t, 128, ActiMode.AC_MODE_RELU)
    t = ffmodel.dense(t, 10)
    t = ffmodel.softmax(t)

    ffoptimizer = SGDOptimizer(ffmodel, 0.01)
    ffmodel.optimizer = ffoptimizer
    ffmodel.compile(loss_type=LossType.LOSS_SPARSE_CATEGORICAL_CROSSENTROPY, metrics=[
                    MetricsType.METRICS_ACCURACY, MetricsType.METRICS_SPARSE_CATEGORICAL_CROSSENTROPY])
    label_tensor = ffmodel.label_tensor

    img_rows, img_cols = 28, 28
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_train = x_train.astype('float32')
    x_train /= 255
    y_train = y_train.astype('int32')
    y_train = np.reshape(y_train, (len(y_train), 1))

    dataloader_input = ffmodel.create_data_loader(input_tensor, x_train)
    dataloader_label = ffmodel.create_data_loader(label_tensor, y_train)

    ffmodel.init_layers()

    epochs = ffconfig.epochs

    ts_start = ffconfig.get_current_time()

    ffmodel.fit(x=dataloader_input, y=dataloader_label, epochs=epochs)

    ts_end = ffconfig.get_current_time()
    run_time = 1e-6 * (ts_end - ts_start)
    print("epochs %d, ELAPSED TIME = %.4fs, THROUGHPUT = %.2f samples/s\n" %
          (epochs, run_time, num_samples * epochs / run_time))

    perf_metrics = ffmodel.get_perf_metrics()

    return perf_metrics


def test_accuracy():
    perf_metrics = top_level_task()
    accuracy = perf_metrics.get_accuracy()
    try:
        assert (accuracy >= ModelAccuracy.MNIST_CNN.value), "Accuracy less than 90%"
    except AssertionError as e:
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--test_acc",
                        action="store_true", help="Test accuracy flag")
    parser.add_argument(
        "-config-file",
        help="The path to a JSON file with the configs. If omitted, a sample model and configs will be used instead.",
        type=str,
        default=None,
    )
    args, unknown = parser.parse_known_args()
    configs_dict = None
    if args.config_file is not None:
        with open(args.config_file) as f:
            configs_dict = json.load(f)
    init_flexflow_runtime(configs_dict)
    if args.test_acc:
        print("Testing mnist cnn training accuracy")
        test_accuracy()
    else:
        print("mnist cnn")
        top_level_task()
