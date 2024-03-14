import flexflow.core
import numpy as np
from flexflow.core import *


def test_residual_rms_norm(
        ffconfig,
        input1_arr: np.ndarray,
        input2_arr: np.ndarray,
        eps: float,
        dim: int,
        name=None,
):
    ffmodel = FFModel(ffconfig)

    input1_tensor = ffmodel.create_tensor(input1_arr.shape, DataType.DT_FLOAT)
    input2_tensor = ffmodel.create_tensor(input2_arr.shape, DataType.DT_FLOAT)

    residual_rms_norm_output1, residual_rms_norm_output2 = ffmodel.residual_rms_norm(
        input1_tensor,
        input2_tensor,
        eps,
        dim,
        name="residual_rms_norm_layer",
    )

    ffoptimizer = SGDOptimizer(ffmodel, 0.001)
    ffmodel.optimizer = ffoptimizer
    ffmodel.compile(
        loss_type=LossType.LOSS_SPARSE_CATEGORICAL_CROSSENTROPY,
        metrics=[MetricsType.METRICS_ACCURACY, MetricsType.METRICS_SPARSE_CATEGORICAL_CROSSENTROPY]
    )

    dataloader_input1 = ffmodel.create_data_loader(input1_tensor, input1_arr)
    dataloader_input2 = ffmodel.create_data_loader(input2_tensor, input2_arr)

    ffmodel.init_layers()

    dataloader_input1.reset()
    dataloader_input1.next_batch(ffmodel)

    dataloader_input2.reset()
    dataloader_input2.next_batch(ffmodel)

    ffmodel.forward()

    residual_rms_norm_output1.inline_map(ffmodel, ffconfig)
    output_result1 = residual_rms_norm_output1.get_array(ffmodel, ffconfig)

    residual_rms_norm_output2.inline_map(ffmodel, ffconfig)
    output_result2 = residual_rms_norm_output2.get_array(ffmodel, ffconfig)

    return output_result1, output_result2


if __name__ == '__main__':
    init_flexflow_runtime()
    ffconfig = FFConfig()

    input1_data = np.random.randn(ffconfig.batch_size, 10, 20).astype(np.float32)
    input2_data = np.random.randn(ffconfig.batch_size, 10, 20).astype(np.float32)
    eps_value = 1e-6
    dim_value = 1  # Example value for dim

    output_result1, output_result2 = test_residual_rms_norm(
        ffconfig,
        input1_data,
        input2_data,
        eps=eps_value,
        dim=dim_value,
    )

    print("Input Array 1:")
    print(input1_data)
    print("\nInput Array 2:")
    print(input2_data)
    print("\nOutput Array 1 after applying residual_rms_norm:")
    print(output_result1)
    print("\nOutput Array 2 after applying residual_rms_norm:")
    print(output_result2)
