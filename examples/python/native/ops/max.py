import flexflow.core
import numpy as np
from flexflow.core import *


def test_max(ffconfig, input_arr1: np.ndarray, input_arr2: np.ndarray) -> np.ndarray:
    ffmodel = FFModel(ffconfig)

    input_tensor1 = ffmodel.create_tensor(input_arr1.shape, DataType.DT_FLOAT)
    input_tensor2 = ffmodel.create_tensor(input_arr2.shape, DataType.DT_FLOAT)

    max_output = ffmodel.max(input_tensor1, input_tensor2, name="max_layer")

    ffoptimizer = SGDOptimizer(ffmodel, 0.001)
    ffmodel.optimizer = ffoptimizer
    ffmodel.compile(
        loss_type=LossType.LOSS_SPARSE_CATEGORICAL_CROSSENTROPY,
        metrics=[MetricsType.METRICS_ACCURACY, MetricsType.METRICS_SPARSE_CATEGORICAL_CROSSENTROPY]
    )

    dataloader_input1 = ffmodel.create_data_loader(input_tensor1, input_arr1)
    dataloader_input2 = ffmodel.create_data_loader(input_tensor2, input_arr2)

    ffmodel.init_layers()

    dataloader_input1.reset()
    dataloader_input2.reset()

    dataloader_input1.next_batch(ffmodel)
    dataloader_input2.next_batch(ffmodel)

    ffmodel.forward()

    max_output.inline_map(ffmodel, ffconfig)
    max_result = max_output.get_array(ffmodel, ffconfig)

    return max_result


if __name__ == '__main__':
    init_flexflow_runtime()
    ffconfig = FFConfig()

    input_data1 = np.random.randn(ffconfig.batch_size, 5, 10, 10).astype(np.float32)
    input_data2 = np.random.randn(ffconfig.batch_size, 5, 10, 10).astype(np.float32)

    max_result = test_max(ffconfig, input_data1, input_data2)

    print("Input Array 1:")
    print(input_data1)
    print("\nInput Array 2:")
    print(input_data2)
    print("\nOutput Array after applying max function:")
    print(max_result)
