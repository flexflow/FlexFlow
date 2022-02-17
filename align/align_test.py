import os
import sys
from typing import Callable

sys.path.append("./align/")

from align_utils import TensorAlignmentData, align_tensors

BASE_DIR = "align/"


def prepend_dirname_fn(dirname: str) -> Callable[[str], str]:
    def f(filename):
        return os.path.join(dirname, filename)
    return f


def test_embedding():
    out_dir = os.path.join(BASE_DIR, "embedding", "out")
    expand = prepend_dirname_fn(out_dir)
    align_tensors(
        [
            TensorAlignmentData(
                "embedding_out",
                expand("ff_out.pt"),
                expand("torch_out.pt"),
            ),
            TensorAlignmentData(
                "embedding_out_grad",
                expand("ff_out_grad.pt"),
                expand("torch_out_grad.pt"),
            ),
            TensorAlignmentData(
                "embedding_weight_grad",
                expand("ff_weight_grad.pt"),
                expand("torch_weight_grad.pt"),
            ),
        ]
    )


def test_layernorm():
    out_dir = os.path.join(BASE_DIR, "layernorm", "out")
    expand = prepend_dirname_fn(out_dir)
    align_tensors(
        [
            TensorAlignmentData(
                "layernorm_out",
                expand("ff_out.pt"),
                expand("torch_out.pt"),
            ),
            TensorAlignmentData(
                "layernorm_out_grad",
                expand("ff_out_grad.pt"),
                expand("torch_out_grad.pt"),
            ),
            TensorAlignmentData(
                "layernorm_weight_grad",
                expand("ff_weight_grad.pt"),
                expand("torch_weight_grad.pt"),
            ),
            TensorAlignmentData(
                "layernorm_bias_grad",
                expand("ff_bias_grad.pt"),
                expand("torch_bias_grad.pt")
            )
        ]
    )


def test_view_embedding():
    out_dir = os.path.join(BASE_DIR, "view_embedding", "out")
    expand = prepend_dirname_fn(out_dir)
    align_tensors(
        [
            TensorAlignmentData(
                "embedding_out",
                expand("ff_out.pt"),
                expand("torch_out.pt"),
            ),
            TensorAlignmentData(
                "embedding_out_grad",
                expand("ff_out_grad.pt"),
                expand("torch_out_grad.pt"),
            ),
            TensorAlignmentData(
                "embedding_weight_grad",
                expand("ff_weight_grad.pt"),
                expand("torch_weight_grad.pt"),
            ),
        ]
    )


def test_getitem():
    out_dir = os.path.join(BASE_DIR, "getitem", "out")
    expand = prepend_dirname_fn(out_dir)
    align_tensors(
        [
            TensorAlignmentData(
                "getitem_out",
                expand("ff_out.pt"),
                expand("torch_out.pt"),
            ),
        ]
    )
