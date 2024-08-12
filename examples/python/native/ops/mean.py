from typing import List

import flexflow.core
import numpy as np
from flexflow.core import *


def test_mean(ffconfig, input_arr: np.ndarray, dims: List[int], keepdims: bool = False) -> np.ndarray:
    ffmodel = FFModel(ffconfig)

    input_tensor = ffmodel.create_tensor(input_arr.shape, DataType.DT_FLOAT)

    mean_output = ffmodel.mean(input_tensor, dims=dims, keepdims=keepdims, name="mean_layer")

    ffoptimizer = SGDOptimizer(ffmodel, 0.001)
    ffmodel.optimizer = ffoptimizer
    ffmodel.compile(
        loss_type=LossType.LOSS_SPARSE_CATEGORICAL_CROSSENTROPY,
        metrics=[MetricsType.METRICS_ACCURACY, MetricsType.METRICS_SPARSE_CATEGORICAL_CROSSENTROPY]
    )

    dataloader_input = ffmodel.create_data_loader(input_tensor, input_arr)

    ffmodel.init_layers()

    dataloader_input.reset()
    dataloader_input.next_batch(ffmodel)
    ffmodel.forward()

    mean_output.inline_map(ffmodel, ffconfig)
    mean_result = mean_output.get_array(ffmodel, ffconfig)

    return mean_result


if __name__ == '__main__':
    init_flexflow_runtime()
    ffconfig = FFConfig()

    input_data = np.random.randn(ffconfig.batch_size, 5, 10, 10).astype(np.float32)
    dims_to_mean = [1, 2]  # Example dimensions to take the mean over

    mean_result = test_mean(ffconfig, input_data, dims=dims_to_mean, keepdims=False)

    print("Input Array:")
    print(input_data)
    print(f"\nOutput Array after applying mean function along dimensions {dims_to_mean}:")
    print(mean_result)
