import os
import sys

import torch

sys.path.append("./align/")
from align_utils import gen_tensor, BATCH_SIZE

assert torch.cuda.is_available(), "Expects at least one GPU"
DEVICE = torch.device(0)
SEQ_LENGTH = 5
OUT_DIR = os.path.join("align", "linear", "out")

def run():
  # define layer in pytorch
  INPUT_SIZE = 512
  OUTPUT_SIZE = 128
  linear = torch.nn.Linear(
      in_features=512,
      out_features=128
  ).to(DEVICE)

  # get weight/bias from ff files, check same shape
  linear_weight = torch.load(os.path.join(OUT_DIR, "ff_weight.pt"))
  linear_bias = torch.load(os.path.join(OUT_DIR, "ff_bias.pt"))
  assert linear.weight.shape == linear_weight.shape, (
      "Shape mismatch: " f"FF={linear_weight.shape} torch={linear.weight.shape}"
  )
  assert linear.bias.shape == linear_bias.shape, (
      "Shape mismatch: " f"FF={linear_bias.shape} torch={linear.bias.shape}"
  )

  # set weight/bias 
  linear.weight = torch.nn.Parameter(linear_weight.to(DEVICE))
  linear.bias = torch.nn.Parameter(linear_bias.to(DEVICE))

  # generate input/label tensors w/ gen_tensor 
  inp: torch.Tensor = gen_tensor(
      (BATCH_SIZE, SEQ_LENGTH, INPUT_SIZE),
      dtype="float32"
  ).to(DEVICE)
  label: torch.Tensor = gen_tensor(
      (BATCH_SIZE, SEQ_LENGTH, OUTPUT_SIZE),
      dtype="float32"
  ).to(DEVICE)

  # get output running input through layer
  output = linear(inp)
  linear.zero_grad()
  output.retain_grad()

  # loss function
  loss_fn = torch.nn.MSELoss(reduction="mean")
  loss = loss_fn(output, label)

  # backpropogate 
  loss.backward()

  # save out, out grad, layer weight & bias gradients
  torch.save(output.cpu(), os.path.join(OUT_DIR, "torch_out.pt"))
  torch.save(output.grad.cpu(), os.path.join(OUT_DIR, "torch_out_grad.pt"))
  torch.save(linear.weight.grad.cpu(), os.path.join(OUT_DIR, "torch_weight_grad.pt"))
  torch.save(linear.bias.grad.cpu(), os.path.join(OUT_DIR, "torch_bias_grad.pt"))


if __name__ == "__main__":
    run()