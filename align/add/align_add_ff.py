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

OUT_DIR = os.path.join("align", "add", "out")


def run():
  INPUT_SIZE = 512
  SEQ_LENGTH = 5
  inp1: torch.Tensor = gen_tensor(
      (BATCH_SIZE, SEQ_LENGTH, INPUT_SIZE),
      dtype="float32"
  )
  inp2: torch.Tensor = gen_tensor(
      (BATCH_SIZE, SEQ_LENGTH, INPUT_SIZE),
      dtype="float32"
  )
  label: torch.Tensor = gen_tensor(
      (BATCH_SIZE, SEQ_LENGTH, INPUT_SIZE),
      dtype="float32"
  )

  ffconfig = FFConfig()
  ffmodel = FFModel(ffconfig)
  input_tensor_1 = ffmodel.create_tensor(inp1.shape, DataType.DT_FLOAT)
  input_tensor_2 = ffmodel.create_tensor(inp2.shape, DataType.DT_FLOAT)
  output_tensor = ffmodel.add(
      x=input_tensor_1,
      y=input_tensor_2,
      name="add"
  )

  # compile 
  compile_ffmodel(ffmodel)
  dls = init_ffmodel(ffmodel, ((input_tensor_1, inp1), (input_tensor_2, inp2)), label)
  assert len(dls) == 3
  inp1_dl, inp2_dl, label_dl = dls

  # forward/backward pass
  run_fwd_bwd(ffmodel, ffconfig, (inp1_dl, inp2_dl), label_dl)

  # save data
  save_tensor_ff(output_tensor, ffmodel, os.path.join(OUT_DIR, "ff_out.pt"))
  save_tensor_grad_ff(output_tensor, ffmodel, os.path.join(OUT_DIR, "ff_out_grad.pt"))

  
  

if __name__ == "__main__":
    run()
