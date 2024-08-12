import flexflow.core
import numpy as np
from flexflow.core import *


def test_sampling(ffconfig, input_arr: np.ndarray, top_p: float, name=None):
    ffmodel = FFModel(ffconfig)

    input_tensor = ffmodel.create_tensor(input_arr.shape, DataType.DT_FLOAT)

    sampling_output = ffmodel.sampling(
        input_tensor,
        top_p,
        name="sampling_layer",
    )

    ffoptimizer = SGDOptimizer(ffmodel, 0.001)
    ffmodel.optimizer = ffoptimizer
    ffmodel.compile(
        loss_type=LossType.LOSS_MEAN_SQUARED_ERROR,
        metrics=[MetricsType.METRICS_MEAN_SQUARED_ERROR],
    )

    dataloader_input = ffmodel.create_data_loader(input_tensor, input_arr)

    ffmodel.init_layers()

    dataloader_input.reset()
    dataloader_input.next_batch(ffmodel)

    ffmodel.forward()

    sampling_output.inline_map(ffmodel, ffconfig)
    output_result = sampling_output.get_array(ffmodel, ffconfig)

    return output_result


if __name__ == '__main__':
    init_flexflow_runtime()
    ffconfig = FFConfig()

    input_data = np.random.randn(ffconfig.batch_size, 10).astype(np.float32)
    top_p_value = 0.8

    output_result = test_sampling(
        ffconfig,
        input_data,
        top_p=top_p_value,
    )

    print("Input Array:")
    print(input_data)
    print("\nOutput Array after applying sampling:")
    print(output_result)
