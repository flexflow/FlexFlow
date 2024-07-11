import flexflow.core
import numpy as np
from flexflow.core import *


def test_argmax(ffconfig, input_arr: np.ndarray, beam_search: bool, name=None):
    ffmodel = FFModel(ffconfig)

    input_tensor = ffmodel.create_tensor(input_arr.shape, DataType.DT_FLOAT)

    argmax_output = ffmodel.argmax(
        input_tensor,
        beam_search,
        name="argmax_layer",
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

    argmax_output.inline_map(ffmodel, ffconfig)
    output_result = argmax_output.get_array(ffmodel, ffconfig)

    return output_result


if __name__ == '__main__':
    init_flexflow_runtime()
    ffconfig = FFConfig()

    input_data = np.random.randn(ffconfig.batch_size, 10).astype(np.float32)
    beam_search_value = True  # Set to True or False based on your requirement

    output_result = test_argmax(
        ffconfig,
        input_data,
        beam_search=beam_search_value,
    )

    print("Input Array:")
    print(input_data)
    print("\nOutput Array after applying argmax:")
    print(output_result)
