import os
import sys

import torch

sys.path.append("./align/")
from align_utils import gen_tensor, BATCH_SIZE

assert torch.cuda.is_available(), "Expects at least one GPU"
DEVICE = torch.device(0)
OUT_DIR = os.path.join("align", "conv2d", "out")

def run():
  KERNEL_SIZE = 3
  INPUT_SIZE = 512
  IN_CHANNELS = 3
  OUTPUT_SIZE = 510
  OUT_CHANNELS = 5
  conv2d = torch.nn.Conv2d(
      in_channels=IN_CHANNELS,
      out_channels=OUT_CHANNELS,
      kernel_size=KERNEL_SIZE
  ).to(DEVICE)

  linear_weight = torch.load(os.path.join(OUT_DIR, "ff_weight.pt"))
  linear_bias = torch.load(os.path.join(OUT_DIR, "ff_bias.pt"))
  assert conv2d.weight.shape == linear_weight.shape, (
      "Shape mismatch: " f"FF={linear_weight.shape} torch={conv2d.weight.shape}"
  )
  assert conv2d.bias.shape == linear_bias.shape, (
      "Shape mismatch: " f"FF={linear_bias.shape} torch={conv2d.bias.shape}"
  )

  conv2d.weight = torch.nn.Parameter(linear_weight.to(DEVICE))
  conv2d.bias = torch.nn.Parameter(linear_bias.to(DEVICE))

  # generate input/label tensors
  # imitating 3-channel image input
  inp: torch.Tensor = gen_tensor(
      (BATCH_SIZE, 3, INPUT_SIZE, INPUT_SIZE),
      dtype="float32"
  ).to(DEVICE)
  label: torch.Tensor = gen_tensor(
      (BATCH_SIZE, 5, OUTPUT_SIZE, OUTPUT_SIZE),
      dtype="float32"
  ).to(DEVICE)

  output = conv2d(inp)
  conv2d.zero_grad()
  output.retain_grad()
  loss_fn = torch.nn.MSELoss(reduction="mean")
  loss = loss_fn(output, label)
  loss.backward()
  torch.save(output.cpu(), os.path.join(OUT_DIR, "torch_out.pt"))
  torch.save(output.grad.cpu(), os.path.join(OUT_DIR, "torch_out_grad.pt"))
  torch.save(conv2d.weight.grad.cpu(), os.path.join(OUT_DIR, "torch_weight_grad.pt"))
  torch.save(conv2d.bias.grad.cpu(), os.path.join(OUT_DIR, "torch_bias_grad.pt"))


if __name__ == "__main__":
    run()