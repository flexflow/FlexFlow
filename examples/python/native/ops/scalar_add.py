import flexflow.core
import numpy as np
from flexflow.core import *


def test_scalar_add(ffconfig, input_arr: np.ndarray, scalar: float, inplace: bool = True, name=None):
    ffmodel = FFModel(ffconfig)

    input_tensor = ffmodel.create_tensor(input_arr.shape, DataType.DT_FLOAT)

    scalar_add_output = ffmodel.scalar_add(
        input_tensor,
        scalar,
        inplace=inplace,
        name="scalar_add_layer"
    )

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

    scalar_add_output.inline_map(ffmodel, ffconfig)
    output_result = scalar_add_output.get_array(ffmodel, ffconfig)

    return output_result


if __name__ == '__main__':
    init_flexflow_runtime()
    ffconfig = FFConfig()

    input_data = np.random.randn(ffconfig.batch_size, 5, 10, 10).astype(np.float32)
    scalar_value = 2.0  # Example scalar value
    inplace_flag = True  # Example inplace flag

    output_result = test_scalar_add(ffconfig, input_data, scalar=scalar_value, inplace=inplace_flag)

    print("Input Array:")
    print(input_data)
    print(f"\nOutput Array after applying scalar addition with scalar value {scalar_value} (inplace={inplace_flag}):")
    print(output_result)
