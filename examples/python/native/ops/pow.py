import flexflow.core
import numpy as np
from flexflow.core import *


def test_pow(ffconfig, input_arr: np.ndarray, exponent: float) -> np.ndarray:
    ffmodel = FFModel(ffconfig)

    input_tensor = ffmodel.create_tensor(input_arr.shape, DataType.DT_FLOAT)

    pow_output = ffmodel.pow(input_tensor, exponent, name="pow_layer")

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

    pow_output.inline_map(ffmodel, ffconfig)
    pow_result = pow_output.get_array(ffmodel, ffconfig)

    return pow_result


if __name__ == '__main__':
    init_flexflow_runtime()
    ffconfig = FFConfig()

    input_data = np.random.randn(ffconfig.batch_size, 5, 10, 10).astype(np.float32)
    exponent_value = 2.0  # Example exponent value

    pow_result = test_pow(ffconfig, input_data, exponent=exponent_value)

    print("Input Array:")
    print(input_data)
    print(f"\nOutput Array after applying pow function with exponent {exponent_value}:")
    print(pow_result)
