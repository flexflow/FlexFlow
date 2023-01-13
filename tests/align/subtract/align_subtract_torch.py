import os
import sys

import torch

sys.path.append("./align/")
from align_utils import gen_tensor, BATCH_SIZE

assert torch.cuda.is_available(), "Expects at least one GPU"
DEVICE = torch.device(0)
OUT_DIR = os.path.join("align", "subtract", "out")

def run():
  INPUT_SIZE = 512
  SEQ_LENGTH = 5

  inp1: torch.Tensor = gen_tensor(
      (BATCH_SIZE, SEQ_LENGTH, INPUT_SIZE),
      dtype="float32"
  ).to(DEVICE)
  inp2: torch.Tensor = gen_tensor(
      (BATCH_SIZE, SEQ_LENGTH, INPUT_SIZE),
      dtype="float32"
  ).to(DEVICE)
  label: torch.Tensor = gen_tensor(
      (BATCH_SIZE, SEQ_LENGTH, INPUT_SIZE),
      dtype="float32"
  ).to(DEVICE)
  output = torch.sub(
      input=inp1,
      other=inp2
  ).to(DEVICE)
  output.requires_grad = True
  output.retain_grad()

  loss_fn = torch.nn.MSELoss(reduction="mean")
  loss = loss_fn(output, label)
  loss.backward()
  torch.save(output.cpu(), os.path.join(OUT_DIR, "torch_out.pt"))
  torch.save(output.grad.cpu(), os.path.join(OUT_DIR, "torch_out_grad.pt"))


if __name__ == "__main__":
    run()