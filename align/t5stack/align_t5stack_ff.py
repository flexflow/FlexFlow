import os
import sys
from collections import OrderedDict

import torch
from flexflow.core import *
from flexflow.core.flexflow_cffi import Embedding, Op, Parameter
from flexflow.torch.model import FunctionNode, PyTorchModel
from flexflow.type import AggrMode

sys.path.append("./align/")


from t5stack.align_t5stack_torch import construct_encoder_t5stack
from align_ff_utils import (compile_ffmodel, init_ffmodel, run_fwd_bwd,
                            save_param_ff, save_param_grad_ff, save_tensor_ff,
                            save_tensor_grad_ff)
from align_utils import BATCH_SIZE, gen_tensor

SEQ_LENGTH = 5
OUT_DIR = "align/t5stack/out/"


def construct_encoder_t5stack_model() -> PyTorchModel:
    t5stack_torch = construct_encoder_t5stack()
    input_names = ["input_ids", "attention_mask"]
    t5stack_model = PyTorchModel(
        t5stack_torch,
        is_hf_model=True,
        input_names=input_names,
        batch_size=BATCH_SIZE,
        seq_length=SEQ_LENGTH,
    )
    return t5stack_model


def gen_inputs_and_labels(num_embeddings, embedding_dim):
    inp: torch.Tensor = gen_tensor(
        (BATCH_SIZE, SEQ_LENGTH),
        dtype="int64",
        low=0,
        high=num_embeddings,
    )
    attention_mask: torch.Tensor = gen_tensor(
        (BATCH_SIZE, SEQ_LENGTH),
        dtype="int64",
        low=0,
        high=2,
    )
    label: torch.Tensor = gen_tensor(
        (BATCH_SIZE, SEQ_LENGTH, embedding_dim),
        dtype="float32",
    )
    return inp, attention_mask, label


def save_ff_params(ffmodel):
    """Saves the FlexFlow model's parameters to be imported to the equivalent
    PyTorch model."""
    # TODO: Move this to align_ff_utils.py
    layers = ffmodel.get_layers()
    model_params = OrderedDict()
    for layer in layers:
        if hasattr(layer, "get_weight_tensor"):
            weight_param: Parameter = layer.get_weight_tensor()
            model_params[f"{layer.name}_weight"] = weight_param
        if hasattr(layer, "get_bias_tensor"):
            bias_param: Parameter = layer.get_bias_tensor()
            model_params[f"{layer.name}_bias"] = bias_param
    torch.save(model_params, os.path.join(OUT_DIR, "ff_model_params.pt"))


def run():
    ffconfig = FFConfig()
    ffmodel = FFModel(ffconfig)

    t5stack_model: PyTorchModel = construct_encoder_t5stack_model()
    num_embeddings = t5stack_model.model.config.vocab_size
    embedding_dim = t5stack_model.model.config.d_model
    inp, attention_mask, label = gen_inputs_and_labels(num_embeddings, embedding_dim)
    input_tensors = [
        ffmodel.create_tensor(inp.shape, DataType.DT_INT64),
        ffmodel.create_tensor(label.shape, DataType.DT_FLOAT),
    ]
    output_tensors, node_to_output = t5stack_model.torch_to_ff(
        ffmodel, input_tensors, verbose=True,
    )
    compile_ffmodel(ffmodel)
    dls = init_ffmodel(
        ffmodel,
        ((input_tensors[0], inp), (input_tensors[1], attention_mask)),
        label,
    )
    assert len(dls) == 3
    # TODO: Disable dropout
    run_fwd_bwd(ffmodel, ffconfig, dls[:-1], dls[-1])

    save_ff_params(ffmodel)
    print(f"len(output_tensors): {len(output_tensors)}")



if __name__ == "__main__":
    run()
