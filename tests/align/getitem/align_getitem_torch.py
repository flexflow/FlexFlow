import os
import sys

import torch

sys.path.append("./align/")
from align_utils import gen_tensor, BATCH_SIZE

assert torch.cuda.is_available(), "Expects at least one GPU"
DEVICE = torch.device(0)
SEQ_LENGTH = 5
OUT_DIR = os.path.join("align", "getitem", "out")


def run():
    """Checks the ``getitem()`` code path for tensor slicing."""
    attention_mask = gen_tensor(
        (BATCH_SIZE, SEQ_LENGTH),
        dtype="int64",
        low=0,
        high=2,
    ).to(DEVICE)
    # Extend to shape (BATCH_SIZE, 1, 1, SEQ_LENGTH)
    extended_attention_mask = attention_mask[:, None, None, :]
    torch.save(extended_attention_mask.cpu(), os.path.join(OUT_DIR, "torch_out.pt"))


if __name__ == "__main__":
    run()
