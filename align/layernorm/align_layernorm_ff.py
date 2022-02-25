import os
import sys

import torch
from flexflow.core import *
from flexflow.core.flexflow_cffi import LayerNorm, Op, Parameter

sys.path.append("./align/")
from align_ff_utils import (compile_ffmodel, init_ffmodel, run_fwd_bwd,
                            save_param_ff, save_param_grad_ff, save_tensor_ff,
                            save_tensor_grad_ff)
from align_utils import gen_tensor, BATCH_SIZE

SEQ_LENGTH = 5
OUT_DIR = os.path.join("align", "layernorm", "out")


def run():
    HIDDEN_SIZE = 512
    EPS = 1e-6
    inp: torch.Tensor = gen_tensor(
        (BATCH_SIZE, SEQ_LENGTH, HIDDEN_SIZE),
        dtype="float32",
    )
    label: torch.Tensor = gen_tensor(
        (BATCH_SIZE, SEQ_LENGTH, HIDDEN_SIZE),
        dtype="float32",
    )

    ffconfig = FFConfig()
    ffmodel = FFModel(ffconfig)
    input_tensor = ffmodel.create_tensor(inp.shape, DataType.DT_FLOAT)
    output_tensor = ffmodel.layer_norm(
        input=input_tensor,
        axes=[len(input_tensor.dims) - 1],  # normalize over the last dimension
        elementwise_affine=True,
        eps=EPS,
        name="layernorm",
    )

    compile_ffmodel(ffmodel)
    dls = init_ffmodel(ffmodel, ((input_tensor, inp),), label)
    assert len(dls) == 2
    inp_dl, label_dl = dls
    run_fwd_bwd(ffmodel, ffconfig, (inp_dl,), label_dl)

    layernorm_layer: Op = ffmodel.get_layers()[0]
    assert isinstance(layernorm_layer, LayerNorm)
    layernorm_weight: Parameter = layernorm_layer.get_weight_tensor()
    layernorm_bias: Parameter = layernorm_layer.get_bias_tensor()
    save_tensor_ff(output_tensor, ffmodel, os.path.join(OUT_DIR, "ff_out.pt"))
    save_tensor_grad_ff(output_tensor, ffmodel, os.path.join(OUT_DIR, "ff_out_grad.pt"))
    save_param_ff(layernorm_weight, ffmodel, os.path.join(OUT_DIR, "ff_weight.pt"))
    save_param_ff(layernorm_bias, ffmodel, os.path.join(OUT_DIR, "ff_bias.pt"))
    save_param_grad_ff(layernorm_weight, ffmodel, os.path.join(OUT_DIR, "ff_weight_grad.pt"))
    save_param_grad_ff(layernorm_bias, ffmodel, os.path.join(OUT_DIR, "ff_bias_grad.pt"))


if __name__ == "__main__":
    run()
