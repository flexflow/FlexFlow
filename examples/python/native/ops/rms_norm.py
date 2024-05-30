import flexflow.core
import numpy as np
from flexflow.core import *


def test_rms_norm(
        ffconfig,
        input_arr: np.ndarray,
        eps: float,
        dim: int,
        name=None,
):
    ffmodel = FFModel(ffconfig)

    input_tensor = ffmodel.create_tensor(input_arr.shape, DataType.DT_FLOAT)

    rms_norm_output = ffmodel.rms_norm(
        input_tensor,
        eps,
        dim,
        name="rms_norm_layer",
    )

    ffoptimizer = SGDOptimizer(ffmodel, 0.001)
    ffmodel.optimizer = ffoptimizer
    ffmodel.compile(
        loss_type=LossType.LOSS_SPARSE_CATEGORICAL_CROSSENTROPY,
        metrics=[MetricsType.METRICS_MEAN_SQUARED_ERROR, MetricsType.METRICS_SPARSE_CATEGORICAL_CROSSENTROPY],
    )

    dataloader_input = ffmodel.create_data_loader(input_tensor, input_arr)

    ffmodel.init_layers()

    dataloader_input.reset()
    dataloader_input.next_batch(ffmodel)

    ffmodel.forward()

    rms_norm_output.inline_map(ffmodel, ffconfig)
    output_result = rms_norm_output.get_array(ffmodel, ffconfig)

    return output_result


if __name__ == '__main__':
    init_flexflow_runtime()
    ffconfig = FFConfig()

    input_data = np.random.randn(ffconfig.batch_size, 10, 20).astype(np.float32)
    eps_value = 1e-6
    dim_value = 1  # Example value for dim

    output_result = test_rms_norm(
        ffconfig,
        input_data,
        eps=eps_value,
        dim=dim_value,
    )

    print("Input Array:")
    print(input_data)
    print("\nOutput Array after applying rms_norm:")
    print(output_result)
