from typing import List

import flexflow.core
import numpy as np
from flexflow.core import *


def test_residual_layer_norm(ffconfig, input_arr: np.ndarray, residual1_arr: np.ndarray, residual2_arr: np.ndarray, use_two_residuals: bool, axes: List[int], elementwise_affine: bool = True, eps: float = 1e-5, use_bias: bool = True, name=None):
    ffmodel = FFModel(ffconfig)

    input_tensor = ffmodel.create_tensor(input_arr.shape, DataType.DT_FLOAT)
    residual1_tensor = ffmodel.create_tensor(residual1_arr.shape, DataType.DT_FLOAT)
    residual2_tensor = ffmodel.create_tensor(residual2_arr.shape, DataType.DT_FLOAT)

    output_tensor, layer_norm_output = ffmodel.residual_layer_norm(
        input_tensor,
        residual1_tensor,
        residual2_tensor if use_two_residuals else None,
        use_two_residuals,
        axes=axes,
        elementwise_affine=elementwise_affine,
        eps=eps,
        use_bias=use_bias,
        name="residual_layer_norm_layer"
    )

    ffoptimizer = SGDOptimizer(ffmodel, 0.001)
    ffmodel.optimizer = ffoptimizer
    ffmodel.compile(
        loss_type=LossType.LOSS_SPARSE_CATEGORICAL_CROSSENTROPY,
        metrics=[MetricsType.METRICS_ACCURACY, MetricsType.METRICS_SPARSE_CATEGORICAL_CROSSENTROPY]
    )

    dataloader_input = ffmodel.create_data_loader(input_tensor, input_arr)
    dataloader_residual1 = ffmodel.create_data_loader(residual1_tensor, residual1_arr)
    dataloader_residual2 = ffmodel.create_data_loader(residual2_tensor, residual2_arr)

    ffmodel.init_layers()

    dataloader_input.reset()
    dataloader_residual1.reset()
    if use_two_residuals:
        dataloader_residual2.reset()

    dataloader_input.next_batch(ffmodel)
    dataloader_residual1.next_batch(ffmodel)
    if use_two_residuals:
        dataloader_residual2.next_batch(ffmodel)

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
    residual1_data = np.random.randn(ffconfig.batch_size, 5, 10, 10).astype(np.float32)
    residual2_data = np.random.randn(ffconfig.batch_size, 5, 10, 10).astype(np.float32)
    use_two_residuals_flag = True  # Example flag

    axes_to_normalize = [1, 2]  # Example axes to normalize

    output_result, layer_norm_result = test_residual_layer_norm(
        ffconfig,
        input_data,
        residual1_data,
        residual2_data,
        use_two_residuals_flag,
        axes=axes_to_normalize,
        elementwise_affine=True,
        eps=1e-5,
        use_bias=True
    )

    print("Input Array:")
    print(input_data)
    print("\nResidual1 Array:")
    print(residual1_data)
    if use_two_residuals_flag:
        print("\nResidual2 Array:")
        print(residual2_data)
    print(f"\nOutput Array after applying residual_layer_norm along axes {axes_to_normalize} with use_two_residuals={use_two_residuals_flag}:")
    print(output_result)
    print("\nLayer Norm Result:")
    print(layer_norm_result)
