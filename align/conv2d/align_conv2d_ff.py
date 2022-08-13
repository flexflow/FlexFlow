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

OUT_DIR = os.path.join("align", "conv2d", "out")


def run():
  KERNEL_SIZE = 3
  INPUT_SIZE = 512
  IN_CHANNELS = 3
  OUTPUT_SIZE = 510
  OUT_CHANNELS = 5
  inp: torch.Tensor = gen_tensor(
      (BATCH_SIZE, IN_CHANNELS, INPUT_SIZE, INPUT_SIZE),
      dtype="float32"
  )
  label: torch.Tensor = gen_tensor(
      (BATCH_SIZE, OUT_CHANNELS, OUTPUT_SIZE, OUTPUT_SIZE),
      dtype="float32"
  )

  ffconfig = FFConfig()
  ffmodel = FFModel(ffconfig)
  input_tensor = ffmodel.create_tensor(inp.shape, DataType.DT_FLOAT)
  output_tensor = ffmodel.conv2d(
      input=input_tensor,
      out_channels=OUT_CHANNELS,
      kernel_h=KERNEL_SIZE, 
      kernel_w=KERNEL_SIZE, 
      stride_h=1, 
      stride_w=1, 
      padding_h=0, 
      padding_w=0,
      name="conv2d"
  )

  # compile model 
  compile_ffmodel(ffmodel)
  dls = init_ffmodel(ffmodel, ((input_tensor, inp),), label)
  assert len(dls) == 2
  inp_dl, label_dl = dls

  # forward/back pass
  run_fwd_bwd(ffmodel, ffconfig, (inp_dl,), label_dl)

  conv2d_layer: Op = ffmodel.get_layers()[0]
  assert isinstance(conv2d_layer, Conv2D)
  conv2d_weight: Parameter = conv2d_layer.get_weight_tensor()
  conv2d_bias: Parameter = conv2d_layer.get_bias_tensor()

  # save output data
  save_tensor_ff(output_tensor, ffmodel, os.path.join(OUT_DIR, "ff_out.pt"))
  save_tensor_grad_ff(output_tensor, ffmodel, os.path.join(OUT_DIR, "ff_out_grad.pt"))
  
  # save layer data
  save_param_ff(conv2d_weight, ffmodel, os.path.join(OUT_DIR, "ff_weight.pt"))
  save_param_ff(conv2d_bias, ffmodel, os.path.join(OUT_DIR, "ff_bias.pt"))
  save_param_grad_ff(conv2d_weight, ffmodel, os.path.join(OUT_DIR, "ff_weight_grad.pt"))
  save_param_grad_ff(conv2d_bias, ffmodel, os.path.join(OUT_DIR, "ff_bias_grad.pt"))

if __name__ == "__main__":
    run()
