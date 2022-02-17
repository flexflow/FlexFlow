import os
import sys
from typing import Tuple

import numpy as np
import torch
from flexflow.core import *
from flexflow.core.flexflow_cffi import (FFConfig, FFModel, Op, Parameter,
                                         SingleDataLoader, Tensor)
from flexflow.type import ParameterSyncType


def ffmodel_barrier(ffmodel):
    # Use `get_current_time()` as a forced sync barrier
    ffmodel._ffconfig.get_current_time()


def compile_ffmodel(ffmodel: FFModel):
    """Compiles the FlexFlow model ``model`` using MSE loss."""
    ffoptimizer = SGDOptimizer(ffmodel, lr=0.01)  # unused
    ffmodel.compile(
        optimizer=ffoptimizer,
        loss_type=LossType.LOSS_MEAN_SQUARED_ERROR_AVG_REDUCE,
        metrics=[MetricsType.METRICS_MEAN_SQUARED_ERROR],
    )


def init_ffmodel(
    ffmodel: FFModel,
    input_tensor: Tensor,
    inp: torch.Tensor,
    label: torch.Tensor,
) -> Tuple[SingleDataLoader, SingleDataLoader]:
    """Initializes the FFModel by creating the data loaders and initializing
    the model layers."""
    inp_dl = ffmodel.create_data_loader(input_tensor, inp.numpy())
    label_dl = ffmodel.create_data_loader(
        ffmodel.label_tensor,
        label.numpy(),
    )
    ffmodel.init_layers()
    return (inp_dl, label_dl)


def run_fwd_bwd(
    ffmodel: FFModel,
    ffconfig: FFConfig,
    inp_dl: SingleDataLoader,
    label_dl: SingleDataLoader,
    run_bwd: bool = True,
) -> None:
    """Runs a single forward pass and backward pass."""
    batch_size = ffconfig.batch_size
    dataloaders = [inp_dl, label_dl]
    num_samples = label_dl.num_samples
    ffmodel._tracing_id += 1
    for d in dataloaders:
        d.reset()
    ffmodel.reset_metrics()
    num_iters = num_samples // batch_size
    assert num_iters == 1, "Internal error: batch size mismatch"
    for d in dataloaders:
        d.next_batch(ffmodel)
    ffmodel._ffconfig.begin_trace(ffmodel._tracing_id)
    ffmodel.forward()
    if run_bwd:
        ffmodel.zero_gradients()
        ffmodel.backward()
    ffmodel._ffconfig.end_trace(ffmodel._tracing_id)
    ffmodel_barrier(ffmodel)


def save_tensor_ff(tensor_ff: Tensor, ffmodel: FFModel, filepath: str) -> None:
    """Saves the FlexFlow tensor ``tensor_ff`` to the filepath ``filepath``."""
    tensor_np: np.ndarray = tensor_ff.get_tensor(ffmodel, ParameterSyncType.PS)
    tensor_torch: torch.Tensor = torch.from_numpy(tensor_np)
    torch.save(tensor_torch, filepath)


def save_tensor_grad_ff(tensor_ff: Tensor, ffmodel: FFModel, filepath: str) -> None:
    """Saves the gradient of the FlexFlow tensor ``tensor_ff`` to the filepath
    ``filepath``."""
    grad_np: np.ndarray = tensor_ff.get_gradients(ffmodel, ParameterSyncType.PS)
    grad_torch: torch.Tensor = torch.from_numpy(grad_np)
    torch.save(grad_torch, filepath)


def save_param_ff(param_ff: Parameter, ffmodel: FFModel, filepath: str) -> None:
    """Saves the FlexFlow parameter ``param_ff`` to the filepath
    ``filepath``."""
    param_np: np.ndarray = param_ff.get_weights(ffmodel)
    param_torch: torch.Tensor = torch.from_numpy(param_np)
    torch.save(param_torch, filepath)


def save_param_grad_ff(param_ff: Parameter, ffmodel: FFModel, filepath: str) -> None:
    """Saves the gradient of the FlexFlow parameter ``param_ff`` to the
    filepath ``filepath``."""
    grad_np: np.ndarray = param_ff.get_gradients(ffmodel, ParameterSyncType.PS)
    grad_torch: torch.Tensor = torch.from_numpy(grad_np)
    torch.save(grad_torch, filepath)
