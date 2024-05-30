import flexflow.core
import numpy as np
from flexflow.core import *


def test_sin(ffconfig, input_arr: np.ndarray) -> np.ndarray:
    ffmodel = FFModel(ffconfig)

    input_tensor = ffmodel.create_tensor(input_arr.shape, DataType.DT_FLOAT)

    sin_output = ffmodel.sin(input_tensor, name="sin_layer")

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

    sin_output.inline_map(ffmodel, ffconfig)
    sin_result = sin_output.get_array(ffmodel, ffconfig)

    return sin_result


if __name__ == '__main__':
    init_flexflow_runtime()
    ffconfig = FFConfig()

    input_data = np.random.randn(ffconfig.batch_size, 5, 10, 10).astype(np.float32)
    sin_result = test_sin(ffconfig, input_data)

    print("Input Array:")
    print(input_data)
    print("\nOutput Array after applying sin function:")
    print(sin_result)
