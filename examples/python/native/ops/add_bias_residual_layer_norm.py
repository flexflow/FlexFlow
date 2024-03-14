from typing import List

import flexflow.core
import numpy as np
from flexflow.core import *


def test_add_bias_residual_layer_norm(ffconfig, input_arr: np.ndarray, residual_arr: np.ndarray, axes: List[int], elementwise_affine: bool = True, eps: float = 1e-5, use_bias: bool = True, name=None):
    ffmodel = FFModel(ffconfig)

    input_tensor = ffmodel.create_tensor(input_arr.shape, DataType.DT_FLOAT)
    residual_tensor = ffmodel.create_tensor(residual_arr.shape, DataType.DT_FLOAT)

    output_tensor, layer_norm_output = ffmodel.add_bias_residual_layer_norm(
        input_tensor,
        residual_tensor,
        axes=axes,
        elementwise_affine=elementwise_affine,
        eps=eps,
        use_bias=use_bias,
        name="add_bias_residual_layer_norm_layer"
    )

    ffoptimizer = SGDOptimizer(ffmodel, 0.001)
    ffmodel.optimizer = ffoptimizer
    ffmodel.compile(
        loss_type=LossType.LOSS_SPARSE_CATEGORICAL_CROSSENTROPY,
        metrics=[MetricsType.METRICS_ACCURACY, MetricsType.METRICS_SPARSE_CATEGORICAL_CROSSENTROPY]
    )

    dataloader_input = ffmodel.create_data_loader(input_tensor, input_arr)
    dataloader_residual = ffmodel.create_data_loader(residual_tensor, residual_arr)

    ffmodel.init_layers()

    dataloader_input.reset()
    dataloader_residual.reset()

    dataloader_input.next_batch(ffmodel)
    dataloader_residual.next_batch(ffmodel)

    ffmodel.forward()

    output_tensor.inline_map(ffmodel, ffconfig)
    layer_norm_output.inline_map(ffmodel, ffconfig)
    output_result = output_tensor.get_array(ffmodel, ffconfig)
    layer_norm_result = layer_norm_output.get_array(ffmodel, ffconfig)

    return output_result, layer_norm_result


if __name__ == '__main__':
    init_flexflow_runtime()
    ffconfig = FFConfig()

    input_data = np.random.randn(ffconfig.batch_size, 5, 10, 10).astype(np.float32)
    residual_data = np.random.randn(ffconfig.batch_size, 5, 10, 10).astype(np.float32)

    axes_to_normalize = [1, 2]  # Example axes to normalize

    output_result, layer_norm_result = test_add_bias_residual_layer_norm(
        ffconfig,
        input_data,
        residual_data,
        axes=axes_to_normalize,
        elementwise_affine=True,
        eps=1e-5,
        use_bias=True
    )

    print("Input Array:")
    print(input_data)
    print("\nResidual Array:")
    print(residual_data)
    print(f"\nOutput Array after applying add_bias_residual_layer_norm along axes {axes_to_normalize}:")
    print(output_result)
    print("\nLayer Norm Result:")
    print(layer_norm_result)
