import flexflow.core
import numpy as np
from flexflow.core import *


def test_arg_top_k(ffconfig, input_arr: np.ndarray, k: int, sorted: bool, speculative_decoding: bool, name=None):
    ffmodel = FFModel(ffconfig)

    input_tensor = ffmodel.create_tensor(input_arr.shape, DataType.DT_FLOAT)

    arg_top_k_output = ffmodel.arg_top_k(
        input_tensor,
        k,
        sorted,
        speculative_decoding,
        name="arg_top_k_layer",
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

    arg_top_k_output.inline_map(ffmodel, ffconfig)
    output_result = arg_top_k_output.get_array(ffmodel, ffconfig)

    return output_result


if __name__ == '__main__':
    init_flexflow_runtime()
    ffconfig = FFConfig()

    input_data = np.random.randn(ffconfig.batch_size, 10).astype(np.float32)
    k_value = 5
    sorted_value = True
    speculative_decoding_value = False  # Example value for speculative_decoding

    output_result = test_arg_top_k(
        ffconfig,
        input_data,
        k=k_value,
        sorted=sorted_value,
        speculative_decoding=speculative_decoding_value,
    )

    print("Input Array:")
    print(input_data)
    print("\nOutput Array after applying arg_top_k:")
    print(output_result)
