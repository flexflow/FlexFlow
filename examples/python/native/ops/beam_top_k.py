import flexflow.core
import numpy as np
from flexflow.core import *


def test_beam_top_k(ffconfig, input_arr: np.ndarray, max_beam_size: int, sorted: bool, name=None):
    ffmodel = FFModel(ffconfig)

    input_tensor = ffmodel.create_tensor(input_arr.shape, DataType.DT_FLOAT)

    beam_top_k_output = ffmodel.beam_top_k(
        input_tensor,
        max_beam_size,
        sorted,
        name="beam_top_k_layer",
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

    beam_top_k_output.inline_map(ffmodel, ffconfig)
    output_result = beam_top_k_output.get_array(ffmodel, ffconfig)

    return output_result


if __name__ == '__main__':
    init_flexflow_runtime()
    ffconfig = FFConfig()

    input_data = np.random.randn(ffconfig.batch_size, 10).astype(np.float32)
    max_beam_size_value = 3
    sorted_value = True

    output_result = test_beam_top_k(
        ffconfig,
        input_data,
        max_beam_size=max_beam_size_value,
        sorted=sorted_value,
    )

    print("Input Array:")
    print(input_data)
    print("\nOutput Array after applying beam_top_k:")
    print(output_result)
