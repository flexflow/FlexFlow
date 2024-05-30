from typing import List

import flexflow.core
import numpy as np
from flexflow.core import *


def test_reduce_sum(ffconfig, input_arr: np.ndarray, axes: List[int], keepdims: bool = False) -> np.ndarray:
    ffmodel = FFModel(ffconfig)

    input_tensor = ffmodel.create_tensor(input_arr.shape, DataType.DT_FLOAT)

    reduce_sum_output = ffmodel.reduce_sum(input_tensor, axes=axes, keepdims=keepdims, name="reduce_sum_layer")

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

    reduce_sum_output.inline_map(ffmodel, ffconfig)
    reduce_sum_result = reduce_sum_output.get_array(ffmodel, ffconfig)

    return reduce_sum_result


if __name__ == '__main__':
    init_flexflow_runtime()
    ffconfig = FFConfig()

    input_data = np.random.randn(ffconfig.batch_size, 5, 10, 10).astype(np.float32)
    axes_to_reduce = [1, 2]  # Example axes to reduce

    reduce_sum_result = test_reduce_sum(ffconfig, input_data, axes=axes_to_reduce, keepdims=False)

    print("Input Array:")
    print(input_data)
    print(f"\nOutput Array after applying reduce_sum along axes {axes_to_reduce}:")
    print(reduce_sum_result)
