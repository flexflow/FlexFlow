import os
import sys

import torch
from flexflow.core import *
from flexflow.core.flexflow_cffi import Linear, Op, Parameter
from flexflow.type import AggrMode

sys.path.append("./align/")
from align_ff_utils import (compile_ffmodel, init_ffmodel, run_fwd_bwd,
                            save_param_ff, save_param_grad_ff, save_tensor_ff,
                            save_tensor_grad_ff)
from align_utils import BATCH_SIZE, gen_tensor

SEQ_LENGTH = 5
OUT_DIR = os.path.join("align", "linear", "out")


def run():
  # create input, label tensors
  INPUT_SIZE = 512
  OUTPUT_SIZE = 128
  inp: torch.Tensor = gen_tensor(
      (BATCH_SIZE, SEQ_LENGTH, INPUT_SIZE),
      dtype="float32"
  )
  label: torch.Tensor = gen_tensor(
      (BATCH_SIZE, SEQ_LENGTH, OUTPUT_SIZE),
      dtype="float32"
  )

  # initialize ffmodel object
  ffconfig = FFConfig()
  ffmodel = FFModel(ffconfig)
  input_tensor = ffmodel.create_tensor(inp.shape, DataType.DT_FLOAT)
  output_tensor = ffmodel.dense(
      input=input_tensor,
      out_dim=128,
      name="linear"
  )


  # compile model 
  compile_ffmodel(ffmodel)

  # fails here
  dls = init_ffmodel(ffmodel, ((input_tensor, inp),), label)
  assert len(dls) == 2
  inp_dl, label_dl = dls

  # forward/back pass
  run_fwd_bwd(ffmodel, ffconfig, (inp_dl,), label_dl)

  # get linear layer
  linear_layer: Op = ffmodel.get_layers()[0]
  assert isinstance(linear_layer, Linear)
  linear_weight: Parameter = linear_layer.get_weight_tensor()
  linear_bias: Parameter = linear_layer.get_bias_tensor()

  # save output data
  save_tensor_ff(output_tensor, ffmodel, os.path.join(OUT_DIR, "ff_out.pt"))
  save_tensor_grad_ff(output_tensor, ffmodel, os.path.join(OUT_DIR, "ff_out_grad.pt"))
  
  # save layer data
  save_param_ff(linear_weight, ffmodel, os.path.join(OUT_DIR, "ff_weight.pt"))
  save_param_ff(linear_bias, ffmodel, os.path.join(OUT_DIR, "ff_bias.pt"))
  save_param_grad_ff(linear_weight, ffmodel, os.path.join(OUT_DIR, "ff_weight_grad.pt"))
  save_param_grad_ff(linear_bias, ffmodel, os.path.join(OUT_DIR, "ff_bias_grad.pt"))

if __name__ == "__main__":
    run()
