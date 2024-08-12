import flexflow.core
import numpy as np
from flexflow.core import *


def test_gather(ffconfig, input_arr: np.ndarray, index_arr: np.ndarray, dim: int, name=None):
    ffmodel = FFModel(ffconfig)

    input_tensor = ffmodel.create_tensor(input_arr.shape, DataType.DT_FLOAT)
    index_tensor = ffmodel.create_tensor(index_arr.shape, DataType.DT_INT32)

    gather_output = ffmodel.gather(
        input_tensor,
        index_tensor,
        dim,
        name="gather_layer"
    )

    ffoptimizer = SGDOptimizer(ffmodel, 0.001)
    ffmodel.optimizer = ffoptimizer
    ffmodel.compile(
        loss_type=LossType.LOSS_SPARSE_CATEGORICAL_CROSSENTROPY,
        metrics=[MetricsType.METRICS_ACCURACY, MetricsType.METRICS_SPARSE_CATEGORICAL_CROSSENTROPY]
    )

    dataloader_input = ffmodel.create_data_loader(input_tensor, input_arr)
    dataloader_index = ffmodel.create_data_loader(index_tensor, index_arr)

    ffmodel.init_layers()

    dataloader_input.reset()
    dataloader_index.reset()

    dataloader_input.next_batch(ffmodel)
    dataloader_index.next_batch(ffmodel)

    ffmodel.forward()

    gather_output.inline_map(ffmodel, ffconfig)
    output_result = gather_output.get_array(ffmodel, ffconfig)

    return output_result


if __name__ == '__main__':
    init_flexflow_runtime()
    ffconfig = FFConfig()

    input_data = np.random.randn(ffconfig.batch_size, 5, 10, 10).astype(np.float32)
    index_data = np.random.randint(0, 5, size=(ffconfig.batch_size,)).astype(np.int32)
    dim_to_gather = 2  # Example dimension to gather along

    output_result = test_gather(ffconfig, input_data, index_data, dim=dim_to_gather)

    print("Input Array:")
    print(input_data)
    print("\nIndex Array:")
    print(index_data)
    print(f"\nOutput Array after applying gather along dimension {dim_to_gather}:")
    print(output_result)
