import os
import sys

import numpy as np
import torch
from flexflow.core import *
from flexflow.core.flexflow_cffi import Embedding, Op, Tensor, Parameter
from flexflow.type import AggrMode, ParameterSyncType

sys.path.append("./align/")
from align_utils import gen_tensor

BATCH_SIZE = 16
SEQ_LENGTH = 5
OUT_DIR = "align/embedding/out/"

def ffmodel_barrier(ffmodel):
    # Use `get_current_time()` as a forced sync barrier
    ffmodel._ffconfig.get_current_time()


def run(backward: bool = False):
    """
    Arguments:
        backward (bool, optional): ``True`` to run the backward pass; ``False``
            otherwise. (Default: ``False``)
    """
    # Create input and label
    num_embeddings = 250112
    embedding_dim = 512
    inp: torch.Tensor = gen_tensor(
        (BATCH_SIZE, SEQ_LENGTH),
        dtype="int64",
        low=0,
        high=num_embeddings,
    )
    label: torch.Tensor = gen_tensor(
        (BATCH_SIZE, SEQ_LENGTH, embedding_dim),
        dtype="float32",
    )
    print(f"[FlexFlow] inp[:16]={inp.flatten()[:16]}")
    print(f"[FlexFlow] label[:16]={label.flatten()[:16]}")

    # Create, compile, and initialize a model consisting of a single embedding
    # layer that uses mean squared error as the loss function
    ffconfig = FFConfig()
    ffmodel = FFModel(ffconfig)
    input_tensor = ffmodel.create_tensor(inp.shape, DataType.DT_INT64)
    embedding_layer_name = "embedding"
    output_tensor = ffmodel.embedding(
        input=input_tensor,
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        aggr=AggrMode.AGGR_MODE_NONE,
        kernel_initializer=NormInitializer(seed=42, mean=0, stddev=1),
        name=embedding_layer_name,
    )
    ffoptimizer = SGDOptimizer(ffmodel, lr=0.01)
    ffmodel.compile(
        optimizer=ffoptimizer,
        loss_type=LossType.LOSS_MEAN_SQUARED_ERROR_AVG_REDUCE,
        metrics=[MetricsType.METRICS_MEAN_SQUARED_ERROR],
    )

    inp_dl = ffmodel.create_data_loader(input_tensor, inp.numpy())
    label_dl = ffmodel.create_data_loader(
        ffmodel.label_tensor,
        label.numpy().astype(np.float32),
    )
    ffmodel.init_layers()

    # NOTE: We simply copy over `ffmodel.fit()`, except now we make the
    # backward pass optional
    # Forward pass
    batch_size = ffconfig.batch_size
    dataloaders = [inp_dl, label_dl]
    num_samples = label_dl.num_samples
    ffmodel._tracing_id += 1
    for d in dataloaders:
        d.reset()
    ffmodel.reset_metrics()
    num_iters = num_samples // batch_size
    assert num_iters == 1
    for d in dataloaders:
        d.next_batch(ffmodel)
    ffmodel._ffconfig.begin_trace(ffmodel._tracing_id)
    ffmodel.forward()

    # Optional backward pass
    if backward:
        ffmodel.zero_gradients()
        ffmodel.backward()

    # Synchronize
    ffmodel._ffconfig.end_trace(ffmodel._tracing_id)
    ffmodel_barrier(ffmodel)

    # Save forward pass output, embedding's weight parameter, and embedding's
    # gradient
    output_np: np.ndarray = output_tensor.get_tensor(ffmodel, ParameterSyncType.PS)
    output_torch: torch.Tensor = torch.from_numpy(output_np)
    print("[FlexFlow] Saving embedding forward pass output...")
    torch.save(output_torch, os.path.join(OUT_DIR, "ff_out.pt"))

    embedding_layer: Op = ffmodel.get_layers()[0]
    assert isinstance(embedding_layer, Embedding)
    embedding_weight: Parameter = embedding_layer.get_weight_tensor()
    embedding_weight_np: np.ndarray = embedding_weight.get_weights(ffmodel)
    embedding_weight_torch: torch.Tensor = torch.from_numpy(embedding_weight_np)
    print("[FlexFlow] Saving embedding weight...")
    torch.save(embedding_weight_torch, os.path.join(OUT_DIR, "ff_embed_weight.pt"))

    if backward:
        weight_grad_np: np.ndarray = embedding_weight.get_gradients(ffmodel, ParameterSyncType.PS)
        weight_grad_torch: torch.Tensor = torch.from_numpy(weight_grad_np)
        print("[FlexFlow] Saving gradient wrt embedding weight...")
        torch.save(weight_grad_torch, os.path.join(OUT_DIR, "ff_weight_grad.pt"))
        out_grad_np: np.ndarray = output_tensor.get_gradients(ffmodel, ParameterSyncType.PS)
        out_grad_torch: torch.Tensor = torch.from_numpy(out_grad_np)
        print("[FlexFlow] Saving gradient wrt embedding output...")
        torch.save(out_grad_torch, os.path.join(OUT_DIR, "ff_out_grad.pt"))


if __name__ == "__main__":
    backward = str(os.environ.get("FF_BACKWARD", True)) in ("True", "1")
    run(backward)
