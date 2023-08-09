import os
import sys

import torch

sys.path.append("./align/")
from align_utils import gen_tensor, BATCH_SIZE

assert torch.cuda.is_available(), "Expects at least one GPU"
DEVICE = torch.device(0)
SEQ_LENGTH = 5
OUT_DIR = os.path.join("align", "view_embedding", "out")


def run():
    NUM_EMBEDDINGS = 250112
    EMBEDDING_DIM = 512
    embedding = torch.nn.Embedding(
        num_embeddings=NUM_EMBEDDINGS,
        embedding_dim=EMBEDDING_DIM,
        device=DEVICE,
    )
    embedding_weight = torch.load(os.path.join(OUT_DIR, "ff_weight.pt"))
    assert embedding_weight.shape == embedding.weight.shape, \
        "Shape mismatch: " \
        f"FF={embedding_weight.shape} torch={embedding.weight.shape}"
    embedding.weight = torch.nn.Parameter(embedding_weight.to(DEVICE))

    inp: torch.Tensor = gen_tensor(
        (BATCH_SIZE, SEQ_LENGTH),
        dtype="int64",
        low=0,
        high=NUM_EMBEDDINGS,
    ).to(DEVICE)
    label: torch.Tensor = gen_tensor(
        (BATCH_SIZE, SEQ_LENGTH, EMBEDDING_DIM),
        dtype="float32",
    ).to(DEVICE)

    output = embedding(inp.view(-1, inp.shape[-1]))
    embedding.zero_grad()
    output.retain_grad()
    loss_fn = torch.nn.MSELoss(reduction="mean")
    loss = loss_fn(output, label)
    loss.backward()

    torch.save(output.cpu(), os.path.join(OUT_DIR, "torch_out.pt"))
    torch.save(output.grad.cpu(), os.path.join(OUT_DIR, "torch_out_grad.pt"))
    torch.save(embedding.weight.grad.cpu(), os.path.join(OUT_DIR, "torch_weight_grad.pt"))


if __name__ == "__main__":
    run()
