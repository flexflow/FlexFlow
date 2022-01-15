import os
import sys

sys.path.append("./align/")

import torch
from align_utils import diff_tensors

OUT_DIR = "align/embedding/out/"


def align_embedding():
    # Check forward pass output
    fwd_out_ff = torch.load(os.path.join(OUT_DIR, "ff_out.pt")).cpu()
    fwd_out_torch = torch.load(os.path.join(OUT_DIR, "torch_out.pt")).cpu()
    diff_tensors(fwd_out_ff, fwd_out_torch)

    # Check gradient


if __name__ == "__main__":
    align_embedding()
