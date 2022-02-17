import argparse
import os
import sys

sys.path.append("./align/")

import torch
from align_utils import diff_tensors

OUT_DIR = "align/embedding/out/"


def align_embedding_fwd():
    fwd_out_ff = torch.load(os.path.join(OUT_DIR, "ff_out.pt")).cpu()
    fwd_out_torch = torch.load(os.path.join(OUT_DIR, "torch_out.pt")).cpu()
    print("[TEST] Checking forward output alignment...")
    try:
        diff_tensors(fwd_out_ff, fwd_out_torch)
        print("[SUCCESS] Forward outputs align!")
    except AssertionError as e:
        print("[FAILURE] Forward outputs did not align!", e)


def align_embedding_bwd():
    # Gradient wrt output
    out_grad_ff = torch.load(os.path.join(OUT_DIR, "ff_out_grad.pt")).cpu()
    out_grad_torch = torch.load(os.path.join(OUT_DIR, "torch_out_grad.pt")).cpu()
    print("[TEST] Checking alignment for gradient wrt embedding output...")
    try:
        diff_tensors(out_grad_ff, out_grad_torch)
        print("[SUCCESS] Gradients align!")
    except AssertionError as e:
        print("[FAILURE] Gradients did not align!", e)

    # Gradient wrt weight
    weight_grad_ff = torch.load(os.path.join(OUT_DIR, "ff_weight_grad.pt")).cpu()
    weight_grad_torch = torch.load(os.path.join(OUT_DIR, "torch_weight_grad.pt")).cpu()
    print("[TEST] Checking alignment for gradient wrt embedding weight...")
    try:
        diff_tensors(weight_grad_ff, weight_grad_torch)
        print("[SUCCESS] Gradients align!")
    except AssertionError as e:
        print("[FAILURE] Gradients did not align!", e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backward", "-b", action="store_true")
    args = parser.parse_args()
    print("=====Embedding Alignment=====")
    align_embedding_fwd()
    if args.backward:
        align_embedding_bwd()
