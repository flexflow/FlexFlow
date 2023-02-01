
import os
import sys

import torch
from flexflow.core import *
from flexflow.core.flexflow_cffi import Linear, Op, Parameter
from flexflow.type import AggrMode

sys.path.append("./align/")
from align_utils import BATCH_SIZE, gen_tensor
from align_ff_utils import (compile_ffmodel, init_ffmodel, run_fwd_bwd,
                            save_param_ff, save_param_grad_ff, save_tensor_ff,
                            save_tensor_grad_ff)

OUT_DIR = os.path.join("align", "pool2d", "out")


def run():
    KERNEL_SIZE = 3
    INPUT_SIZE = 512
    IN_CHANNELS = 3
    OUTPUT_SIZE = 510
    inp: torch.Tensor = gen_tensor(
        (BATCH_SIZE, IN_CHANNELS, INPUT_SIZE, INPUT_SIZE),
        dtype="float32"
    )
    label: torch.Tensor = gen_tensor(
        (BATCH_SIZE, IN_CHANNELS, OUTPUT_SIZE, OUTPUT_SIZE),
        dtype="float32"
    )

    ffconfig = FFConfig()
    ffmodel = FFModel(ffconfig)
    input_tensor = ffmodel.create_tensor(inp.shape, DataType.DT_FLOAT)
    
    output_tensor = ffmodel.pool2d(
        input=input_tensor,
        kernel_h=KERNEL_SIZE,
        kernel_w=KERNEL_SIZE,
        stride_h=1,
        stride_w=1,
        padding_h=0,
        padding_w=0,
        name="pool2d"
    )

    # compile model
    compile_ffmodel(ffmodel)
    dls = init_ffmodel(ffmodel, ((input_tensor, inp),), label)
    assert len(dls) == 2
    inp_dl, label_dl = dls

    # forward/back pass
    run_fwd_bwd(ffmodel, ffconfig, (inp_dl,), label_dl)

    # save output data
    save_tensor_ff(output_tensor, ffmodel, os.path.join(OUT_DIR, "ff_out.pt"))
    save_tensor_grad_ff(output_tensor, ffmodel,
                        os.path.join(OUT_DIR, "ff_out_grad.pt"))

if __name__ == "__main__":
    run()
