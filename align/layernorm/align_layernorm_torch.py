import os
import sys

import torch

sys.path.append("./align/")
from align_utils import gen_tensor, BATCH_SIZE

assert torch.cuda.is_available(), "Expects at least one GPU"
DEVICE = torch.device(0)
SEQ_LENGTH = 5
OUT_DIR = os.path.join("align", "layernorm", "out")


def run():
    HIDDEN_SIZE = 512
    EPS = 1e-6
    layernorm = torch.nn.LayerNorm(
        normalized_shape=HIDDEN_SIZE,
        eps=EPS,
        elementwise_affine=True,
    ).to(DEVICE)
    layernorm_weight = torch.load(os.path.join(OUT_DIR, "ff_weight.pt"))
    layernorm_bias = torch.load(os.path.join(OUT_DIR, "ff_bias.pt"))
    assert layernorm.weight.shape == layernorm_weight.shape, (
        "Shape mismatch: " f"FF={layernorm_weight.shape} torch={layernorm.weight.shape}"
    )
    assert layernorm.bias.shape == layernorm_bias.shape, (
        "Shape mismatch: " f"FF={layernorm_bias.shape} torch={layernorm.bias.shape}"
    )
    layernorm.weight = torch.nn.Parameter(layernorm_weight.to(DEVICE))
    layernorm.bias = torch.nn.Parameter(layernorm_bias.to(DEVICE))

    inp: torch.Tensor = gen_tensor(
        (BATCH_SIZE, SEQ_LENGTH, HIDDEN_SIZE),
        dtype="float32",
    ).to(DEVICE)
    label: torch.Tensor = gen_tensor(
        (BATCH_SIZE, SEQ_LENGTH, HIDDEN_SIZE),
        dtype="float32",
    ).to(DEVICE)

    output = layernorm(inp)
    layernorm.zero_grad()
    output.retain_grad()
    loss_fn = torch.nn.MSELoss(reduction="mean")
    loss = loss_fn(output, label)
    loss.backward()
    torch.save(output.cpu(), os.path.join(OUT_DIR, "torch_out.pt"))
    torch.save(output.grad.cpu(), os.path.join(OUT_DIR, "torch_out_grad.pt"))
    torch.save(layernorm.weight.grad.cpu(), os.path.join(OUT_DIR, "torch_weight_grad.pt"))
    torch.save(layernorm.bias.grad.cpu(), os.path.join(OUT_DIR, "torch_bias_grad.pt"))


if __name__ == "__main__":
    run()
