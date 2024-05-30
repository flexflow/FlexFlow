from flexflow.core import *
from flexflow.keras.datasets import cifar10

from accuracy import ModelAccuracy
from PIL import Image
import argparse, json
import numpy as np


def top_level_task():
    ffconfig = FFConfig()
    alexnetconfig = NetConfig()
    print(alexnetconfig.dataset_path)
    print("Python API batchSize(%d) workersPerNodes(%d) numNodes(%d)" % (
        ffconfig.batch_size, ffconfig.workers_per_node, ffconfig.num_nodes))
    ffmodel = FFModel(ffconfig)

    dims_input = [ffconfig.batch_size, 3, 229, 229]
    input_tensor = ffmodel.create_tensor(dims_input, DataType.DT_FLOAT)

    kernel_init = GlorotUniformInitializer(123)
    bias_init = ZeroInitializer()
    t = ffmodel.conv2d(input_tensor, 64, 11, 11, 4, 4, 2, 2,
                       ActiMode.AC_MODE_RELU, 1, True, None, kernel_init, bias_init)
    t = ffmodel.pool2d(t, 3, 3, 2, 2, 0, 0)
    t = ffmodel.conv2d(t, 192, 5, 5, 1, 1, 2, 2, ActiMode.AC_MODE_RELU)
    t = ffmodel.pool2d(t, 3, 3, 2, 2, 0, 0)
    t = ffmodel.conv2d(t, 384, 3, 3, 1, 1, 1, 1, ActiMode.AC_MODE_RELU)
    t = ffmodel.conv2d(t, 256, 3, 3, 1, 1, 1, 1, ActiMode.AC_MODE_RELU)
    t = ffmodel.conv2d(t, 256, 3, 3, 1, 1, 1, 1, ActiMode.AC_MODE_RELU)
    t = ffmodel.pool2d(t, 3, 3, 2, 2, 0, 0)
    t = ffmodel.flat(t)
    t = ffmodel.dense(t, 4096, ActiMode.AC_MODE_RELU)
    t = ffmodel.dense(t, 4096, ActiMode.AC_MODE_RELU)
    t = ffmodel.dense(t, 10)
    t = ffmodel.softmax(t)

    ffoptimizer = SGDOptimizer(ffmodel, 0.01)
    ffmodel.optimizer = ffoptimizer
    ffmodel.compile(loss_type=LossType.LOSS_SPARSE_CATEGORICAL_CROSSENTROPY, metrics=[
                    MetricsType.METRICS_ACCURACY, MetricsType.METRICS_SPARSE_CATEGORICAL_CROSSENTROPY])
    label_tensor = ffmodel.label_tensor

    num_samples = 10000

    (x_train, y_train), (x_test, y_test) = cifar10.load_data(num_samples)

    full_input_np = np.zeros((num_samples, 3, 229, 229), dtype=np.float32)

    for i in range(0, num_samples):
        image = x_train[i, :, :, :]
        image = image.transpose(1, 2, 0)
        pil_image = Image.fromarray(image)
        pil_image = pil_image.resize((229, 229), Image.Resampling.NEAREST)
        image = np.array(pil_image, dtype=np.float32)
        image = image.transpose(2, 0, 1)
        full_input_np[i, :, :, :] = image
        if (i == 0):
            print(image)

    full_input_np /= 255
    print(full_input_np.shape)
    print(full_input_np.__array_interface__["strides"])
    print(full_input_np[0, :, :, :])

    y_train = y_train.astype('int32')
    full_label_np = y_train

    dataloader_input = ffmodel.create_data_loader(input_tensor, full_input_np)
    dataloader_label = ffmodel.create_data_loader(label_tensor, full_label_np)

    num_samples = dataloader_input.num_samples
    assert dataloader_input.num_samples == dataloader_label.num_samples

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
    accuracy = perf_metrics.get_accuracy()
    if accuracy < ModelAccuracy.CIFAR10_ALEXNET.value:
        assert 0, 'Check Accuracy'

    # conv_2d1 = ffmodel.get_layer_by_id(0)
    # cbias_tensor = conv_2d1.get_input_tensor()
    # cbias_tensor.inline_map(ffconfig)
    # cbias = cbias_tensor.get_flat_array(ffconfig, DataType.DT_FLOAT)
    # print(cbias.shape)
    # print(cbias)
    # #save_image(cbias, 2)
    # cbias_tensor.inline_unmap(ffconfig)

    # label.inline_map(ffconfig)
    # label_array = label.get_flat_array(ffconfig, DataType.DT_INT32)
    # print(label_array.shape)
    # # print(cbias)
    # print(label_array)
    # label.inline_unmap(ffconfig)

    # ffmodel.print_layers(0)


def save_image(batch_image_array, id):
    image_array = batch_image_array[id, :, :, :]
    image_array = image_array.transpose(1, 2, 0)
    image_array = image_array*255
    image_array = image_array.astype('uint8')
    pil_image = Image.fromarray(image_array).convert('RGB')
    pil_image.save("img.jpeg")


def test_accuracy():
    perf_metrics = top_level_task()
    accuracy = perf_metrics.get_accuracy()
    try:
        assert (
            accuracy >= ModelAccuracy.CIFAR10_ALEXNET.value), "Accuracy less than 90%"
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
        print("Testing cifar10 alexnet training accuracy")
        test_accuracy()
    else:
        print("cifar10 alexnet")
        top_level_task()
