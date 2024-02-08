import flexflow.core
import numpy as np
from flexflow.core import *


def test_cos(ffconfig, input_arr: np.ndarray) -> np.ndarray:
    ffmodel = FFModel(ffconfig)

    input_tensor = ffmodel.create_tensor(input_arr.shape, DataType.DT_FLOAT)

    cos_output = ffmodel.cos(input_tensor, name="cos_layer")

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

    cos_output.inline_map(ffmodel, ffconfig)
    cos_result = cos_output.get_array(ffmodel, ffconfig)

    return cos_result


if __name__ == '__main__':
    init_flexflow_runtime()
    ffconfig = FFConfig()

    input_data = np.random.randn(ffconfig.batch_size, 5, 10, 10).astype(np.float32)
    cos_result = test_cos(ffconfig, input_data)

    print("Input Array:")
    print(input_data)
    print("\nOutput Array after applying cos function:")
    print(cos_result)
