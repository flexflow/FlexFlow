import os
import sys

import torch

sys.path.append("./align/")
from align_utils import gen_tensor, BATCH_SIZE


assert torch.cuda.is_available(), "Expects at least one GPU"
DEVICE = torch.device(0)
OUT_DIR = os.path.join("align", "pool2d", "out")


def run():
    KERNEL_SIZE = 3
    INPUT_SIZE = 512
    IN_CHANNELS = 3
    OUTPUT_SIZE = 510

    

    
    # generate input/label tensors
    # imitating 3-channel image input
    inp: torch.Tensor = gen_tensor(
        (BATCH_SIZE, IN_CHANNELS, INPUT_SIZE, INPUT_SIZE),
        dtype="float32"
    ).to(DEVICE)
    label: torch.Tensor = gen_tensor(
        (BATCH_SIZE, IN_CHANNELS, OUTPUT_SIZE, OUTPUT_SIZE),
        dtype="float32"
    ).to(DEVICE)
    pool2d = torch.nn.MaxPool2d(kernel_size=KERNEL_SIZE, stride=1, padding=0).to(DEVICE)

    
    output = pool2d(inp)
    output.requires_grad = True
    pool2d.zero_grad()
    output.retain_grad()
    loss_fn = torch.nn.MSELoss(reduction="mean")
    
    loss = loss_fn(output, label)
    loss.backward()
    torch.save(output.cpu(), os.path.join(OUT_DIR, "torch_out.pt"))
    torch.save(output.grad.cpu(), os.path.join(OUT_DIR, "torch_out_grad.pt"))


if __name__ == "__main__":
    run()
