from typing import List

import flexflow.core
import numpy as np
from flexflow.core import *


def test_layer_norm(ffconfig, input_arr: np.ndarray, axes: List[int], elementwise_affine: bool = True, eps: float = 1e-5, use_bias: bool = True, name=None) -> np.ndarray:
    ffmodel = FFModel(ffconfig)

    input_tensor = ffmodel.create_tensor(input_arr.shape, DataType.DT_FLOAT)

    layer_norm_output = ffmodel.layer_norm(input_tensor, axes=axes, elementwise_affine=elementwise_affine, eps=eps, use_bias=use_bias, name="layer_norm_layer")

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

    layer_norm_output.inline_map(ffmodel, ffconfig)
    layer_norm_result = layer_norm_output.get_array(ffmodel, ffconfig)

    return layer_norm_result


if __name__ == '__main__':
    init_flexflow_runtime()
    ffconfig = FFConfig()

    input_data = np.random.randn(ffconfig.batch_size, 5, 10, 10).astype(np.float32)
    axes_to_normalize = [1, 2]  # Example axes to normalize

    layer_norm_result = test_layer_norm(ffconfig, input_data, axes=axes_to_normalize, elementwise_affine=True, eps=1e-5, use_bias=True)

    print("Input Array:")
    print(input_data)
    print(f"\nOutput Array after applying layer_norm function along axes {axes_to_normalize}:")
    print(layer_norm_result)
