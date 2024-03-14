import flexflow.core
import numpy as np
from flexflow.core import *


def test_sigmoid_silu_multi(ffconfig, input1_arr: np.ndarray, input2_arr: np.ndarray, name=None):
    ffmodel = FFModel(ffconfig)

    input1_tensor = ffmodel.create_tensor(input1_arr.shape, DataType.DT_FLOAT)
    input2_tensor = ffmodel.create_tensor(input2_arr.shape, DataType.DT_FLOAT)

    sigmoid_silu_multi_output = ffmodel.sigmoid_silu_multi(
        input1_tensor,
        input2_tensor,
        name="sigmoid_silu_multi_layer"
    )

    ffoptimizer = SGDOptimizer(ffmodel, 0.001)
    ffmodel.optimizer = ffoptimizer
    ffmodel.compile(
        loss_type=LossType.LOSS_SPARSE_CATEGORICAL_CROSSENTROPY,
        metrics=[MetricsType.METRICS_ACCURACY, MetricsType.METRICS_SPARSE_CATEGORICAL_CROSSENTROPY]
    )

    dataloader_input1 = ffmodel.create_data_loader(input1_tensor, input1_arr)
    dataloader_input2 = ffmodel.create_data_loader(input2_tensor, input2_arr)

    ffmodel.init_layers()

    dataloader_input1.reset()
    dataloader_input2.reset()

    dataloader_input1.next_batch(ffmodel)
    dataloader_input2.next_batch(ffmodel)

    ffmodel.forward()

    sigmoid_silu_multi_output.inline_map(ffmodel, ffconfig)
    output_result = sigmoid_silu_multi_output.get_array(ffmodel, ffconfig)

    return output_result


if __name__ == '__main__':
    init_flexflow_runtime()
    ffconfig = FFConfig()

    input1_data = np.random.randn(ffconfig.batch_size, 5, 10, 10).astype(np.float32)
    input2_data = np.random.randn(ffconfig.batch_size, 5, 10, 10).astype(np.float32)

    output_result = test_sigmoid_silu_multi(ffconfig, input1_data, input2_data)

    print("Input1 Array:")
    print(input1_data)
    print("\nInput2 Array:")
    print(input2_data)
    print("\nOutput Array after applying sigmoid_silu_multi:")
    print(output_result)
