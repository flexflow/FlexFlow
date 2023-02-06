import argparse
import os
import sys

import torch

sys.path.append("./align/")
from align_utils import gen_tensor

assert torch.cuda.is_available(), "Expects at least one GPU"
DEVICE = torch.device(0)
BATCH_SIZE = 16
SEQ_LENGTH = 5
OUT_DIR = os.path.join("align", "layernorm", "out")


class T5LayerNorm(torch.nn.Module):
    """See https://github.com/huggingface/transformers/blob/master/src/transformers/models/t5/modeling_t5.py"""

    def __init__(self, hidden_size, eps=1e-6):
        """Construct a layernorm module in the T5 style (no bias and no
        subtraction of mean)."""
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)
        return self.weight * hidden_states


def run():
    # Initialize the T5 layer norm and load the weight from FlexFlow
    HIDDEN_SIZE = 512
    t5_layernorm = T5LayerNorm(HIDDEN_SIZE).to(DEVICE)
    t5_layernorm_weight = torch.load(os.path.join(OUT_DIR, "ff_layernorm_weight.pt"))
    assert t5_layernorm.weight.shape == t5_layernorm_weight.shape, (
        "Shape mismatch: "
        f"FF={t5_layernorm_weight.shape} torch={t5_layernorm.weight.shape}"
    )
    t5_layernorm.weight = torch.nn.Parameter(t5_layernorm_weight.to(DEVICE))

    inp: torch.Tensor = gen_tensor(
        (BATCH_SIZE, SEQ_LENGTH, HIDDEN_SIZE),
        dtype="float32",
    ).to(DEVICE)
    label: torch.Tensor = gen_tensor(
        (BATCH_SIZE, SEQ_LENGTH, HIDDEN_SIZE),
        dtype="float32",
    ).to(DEVICE)

    output = t5_layernorm(inp)
    torch.save(output.cpu(), os.path.join(OUT_DIR, "torch_out.pt"))

    t5_layernorm.zero_grad()
    output.retain_grad()
    loss_fn = torch.nn.MSELoss(reduction="mean")
    loss = loss_fn(output, label)
    loss.backward()
    torch.save(
        t5_layernorm.weight.grad.cpu(), os.path.join(OUT_DIR, "torch_weight_grad.pt")
    )
    torch.save(output.grad.cpu(), os.path.join(OUT_DIR, "torch_out_grad.pt"))


if __name__ == "__main__":
    run()
