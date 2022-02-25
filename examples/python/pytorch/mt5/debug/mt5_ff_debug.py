import os
import sys

import numpy as np
import torch
from flexflow.core import *
from flexflow.torch.model import PyTorchModel
from flexflow.type import ParameterSyncType
from transformers import MT5ForConditionalGeneration

sys.path.append("examples/python/pytorch/mt5/debug")
from mt5_ff_utils import init_ff_mt5


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def train_step_ff():
    ffmodel, node_to_output, input_dls, labels_dl = init_ff_mt5()

    print("Training...")
    # ffmodel.fit(
    #     x=[input_ids_dl, attention_mask_dl, decoder_input_ids_dl],
    #     y=labels_dl, batch_size=batch_size, epochs=1,
    # )
    x = input_dls
    y = labels_dl
    dataloaders = x + [y]
    num_samples = y.num_samples
    ffmodel._tracing_id += 1
    for d in dataloaders:
        d.reset()
    ffmodel.reset_metrics()
    batch_size = ffmodel._ffconfig.batch_size
    num_iters = num_samples // batch_size
    assert num_iters == 1
    for d in dataloaders:
        d.next_batch(ffmodel)
    ffmodel._ffconfig.begin_trace(ffmodel._tracing_id)
    ffmodel.forward()
    ffmodel.zero_gradients()
    ffmodel.backward()
    ffmodel.update()
    ffmodel._ffconfig.end_trace(ffmodel._tracing_id)
    ffmodel._ffconfig.get_current_time()  # synchronization barrier

    # Print per-layer information
    # for i, node in enumerate(node_to_output):
    #     layer = ffmodel.get_layer_by_name(node)
    #     if layer is not None and hasattr(layer, "get_output_tensor"):
    #         np_array = layer.get_output_tensor().get_tensor(ffmodel, ParameterSyncType.PS)
    #         # print(f"{node}\t{np.linalg.norm(np_array):.3f}")
    #         # print(f"{node}\t{np_array}\n")
    #         print(f"{i}: {node}")
    #         if i > 5:
    #             break


if __name__ == "__main__":
    train_step_ff()
