# AI generated from conv2d example
import flexflow.core
import numpy as np
from flexflow.core import *


def test_pool2d(ffconfig, input_arr: np.ndarray) -> flexflow.core.Tensor:
    ffmodel = FFModel(ffconfig)

    input_tensor = ffmodel.create_tensor(input_arr.shape, DataType.DT_FLOAT)

    out = ffmodel.pool2d(input_tensor, 3, 3, 1, 1, 0, 0, PoolType.POOL_MAX)

    ffoptimizer = SGDOptimizer(ffmodel, 0.001)
    ffmodel.optimizer = ffoptimizer
    ffmodel.compile(
        loss_type=LossType.LOSS_SPARSE_CATEGORICAL_CROSSENTROPY,
        metrics=[MetricsType.METRICS_ACCURACY, MetricsType.METRICS_SPARSE_CATEGORICAL_CROSSENTROPY])
    dataloader_input = ffmodel.create_data_loader(input_tensor, input_arr)

    ffmodel.init_layers()

    dataloader_input.reset()
    dataloader_input.next_batch(ffmodel)
    ffmodel.forward()

    out.inline_map(ffmodel, ffconfig)
    return out.get_array(ffmodel, ffconfig)


if __name__ == '__main__':
    init_flexflow_runtime()
    ffconfig = FFConfig()

    input = np.random.randn(ffconfig.batch_size, 5, 10, 10).astype(np.float32)
    _ = test_pool2d(ffconfig, input)