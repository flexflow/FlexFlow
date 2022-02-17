import argparse
import os
import sys

import torch

sys.path.append("./align/")
from align_utils import gen_tensor

assert torch.cuda.is_available(), "Expects at least one GPU"
DEVICE = torch.device(0)
# DEVICE = torch.device("cpu")
BATCH_SIZE = 16
SEQ_LENGTH = 5
OUT_DIR = "align/embedding/out/"
PRINT_LIMIT = 17


def run(backward: bool = False, verbose: bool = False):
    """
    Arguments:
        backward (bool, optional): ``True`` to run the backward pass; ``False``
            otherwise. (Default: ``False``)
    """
    # Initialize the embedding layer and load the weight from FlexFlow
    num_embeddings = 250112
    embedding_dim = 512
    embedding = torch.nn.Embedding(
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        device=DEVICE,
    )
    print("[PyTorch] Loading embedding weight from FlexFlow")
    embedding_weight = torch.load(os.path.join(OUT_DIR, "ff_embed_weight.pt"))
    assert embedding_weight.shape == embedding.weight.shape, \
        "Shape mismatch: " \
        f"FF={embedding_weight.shape} torch={embedding.weight.shape}"
    embedding.weight = torch.nn.Parameter(embedding_weight.to(DEVICE))

    inp: torch.Tensor = gen_tensor(
        (BATCH_SIZE, SEQ_LENGTH),
        dtype="int64",
        low=0,
        high=num_embeddings,
    ).to(DEVICE)
    label: torch.Tensor = gen_tensor(
        (BATCH_SIZE, SEQ_LENGTH, embedding_dim),
        dtype="float32",
    ).to(DEVICE)
    print(f"[PyTorch] inp[:16]={inp.flatten()[:16]}")
    print(f"[PyTorch] label[:16]={label.flatten()[:16]}")

    # Forward pass
    output = embedding(inp)
    print("[PyTorch] Saving emebedding forward pass output...")
    torch.save(output, os.path.join(OUT_DIR, "torch_out.pt"))

    # Optional information print
    if verbose:
        _weight = embedding.weight
        print(
            f"[embedding weight] {_weight.shape}\n"
            f"{_weight.flatten()[:PRINT_LIMIT]}"
        )
        print(
            f"[embedding output] {output.shape}\n"
            f"{output.flatten()[:PRINT_LIMIT]}"
        )

    # Optional backward pass
    if backward:
        embedding.zero_grad()
        output.retain_grad()  # save the gradient wrt embedding output
        loss_fn = torch.nn.MSELoss(reduction="mean")
        loss = loss_fn(output, label)
        print(f"[PyTorch] loss={loss}")
        loss.backward()
        print("[PyTorch] Saving gradient wrt embedding weight...")
        torch.save(embedding.weight.grad, os.path.join(OUT_DIR, "torch_weight_grad.pt"))
        print("[PyTorch] Saving gradient wrt embedding output...")
        torch.save(output.grad, os.path.join(OUT_DIR, "torch_out_grad.pt"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backward", "-b", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()
    run(backward=args.backward, verbose=args.verbose)
