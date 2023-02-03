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


def test_conv2d():
    out_dir = os.path.join(BASE_DIR, "conv2d", "out")
    expand = prepend_dirname_fn(out_dir)
    align_tensors(
        [
            TensorAlignmentData(
                "conv2d_out",
                expand("ff_out.pt"),
                expand("torch_out.pt"),
            ),
            TensorAlignmentData(
                "conv2d_out_grad",
                expand("ff_out_grad.pt"),
                expand("torch_out_grad.pt"),
            ),
            TensorAlignmentData(
                "conv2d_weight_grad",
                expand("ff_weight_grad.pt"),
                expand("torch_weight_grad.pt"),
            ),
            TensorAlignmentData(
                "conv2d_bias_grad",
                expand("ff_bias_grad.pt"),
                expand("torch_bias_grad.pt")
            )
          ]
    )


def test_add():
    out_dir = os.path.join(BASE_DIR, "add", "out")
    expand = prepend_dirname_fn(out_dir)
    align_tensors(
        [
            TensorAlignmentData(
                "add_out",
                expand("ff_out.pt"),
                expand("torch_out.pt"),
            ),
            TensorAlignmentData(
                "add_out_grad",
                expand("ff_out_grad.pt"),
                expand("torch_out_grad.pt"),
            ),
        ]
    )
def test_concat():
    out_dir = os.path.join(BASE_DIR, "concat", "out")
    expand = prepend_dirname_fn(out_dir)
    align_tensors(
        [
            TensorAlignmentData(
                "concat_out",
                expand("ff_out.pt"),
                expand("torch_out.pt"),
            ),
            TensorAlignmentData(
                "concat_out_grad",
                expand("ff_out_grad.pt"),
                expand("torch_out_grad.pt"),
            ),
        ]
    )

def test_subtract():
    out_dir = os.path.join(BASE_DIR, "subtract", "out")
    expand = prepend_dirname_fn(out_dir)
    align_tensors(
        [
            TensorAlignmentData(
                "subtract_out",
                expand("ff_out.pt"),
                expand("torch_out.pt"),
            ),
            TensorAlignmentData(
                "subtract_out_grad",
                expand("ff_out_grad.pt"),
                expand("torch_out_grad.pt"),
            ),
        ]
    )

def test_multiply():
    out_dir = os.path.join(BASE_DIR, "multiply", "out")
    expand = prepend_dirname_fn(out_dir)
    align_tensors(
        [
            TensorAlignmentData(
                "multiply_out",
                expand("ff_out.pt"),
                expand("torch_out.pt"),
            ),
            TensorAlignmentData(
                "multiply_out_grad",
                expand("ff_out_grad.pt"),
                expand("torch_out_grad.pt"),
            ),
        ]
    )
    
def test_pool2d():
    out_dir = os.path.join(BASE_DIR, "pool2d", "out")
    expand = prepend_dirname_fn(out_dir)
    align_tensors(
        [
            TensorAlignmentData(
                "pool2d_out",
                expand("ff_out.pt"),
                expand("torch_out.pt"),
            ),
            TensorAlignmentData(
                "pool2d_out_grad",
                expand("ff_out_grad.pt"),
                expand("torch_out_grad.pt"),
            ),
        ]
    )
    
def test_reducesum():
    out_dir = os.path.join(BASE_DIR, "reducesum", "out")
    expand = prepend_dirname_fn(out_dir)
    align_tensors(
        [
            TensorAlignmentData(
                "reducesum_out",
                expand("ff_out.pt"),
                expand("torch_out.pt"),
            ),
            TensorAlignmentData(
                "reducesum_out_grad",
                expand("ff_out_grad.pt"),
                expand("torch_out_grad.pt"),
            ),
        ]
    )
    
def test_reshape():
    out_dir = os.path.join(BASE_DIR, "reshape", "out")
    expand = prepend_dirname_fn(out_dir)
    align_tensors(
        [
            TensorAlignmentData(
                "reshape_out",
                expand("ff_out.pt"),
                expand("torch_out.pt"),
            ),
            TensorAlignmentData(
                "reshape_out_grad",
                expand("ff_out_grad.pt"),
                expand("torch_out_grad.pt"),
            ),
        ]
    )
    
def test_flat():
    out_dir = os.path.join(BASE_DIR, "flat", "out")
    expand = prepend_dirname_fn(out_dir)
    align_tensors(
        [
            TensorAlignmentData(
                "flat_out",
                expand("ff_out.pt"),
                expand("torch_out.pt"),
            ),
            TensorAlignmentData(
                "flat_out_grad",
                expand("ff_out_grad.pt"),
                expand("torch_out_grad.pt"),
            ),
        ]
    )
    
def test_sin():
    out_dir = os.path.join(BASE_DIR, "sin", "out")
    expand = prepend_dirname_fn(out_dir)
    align_tensors(
        [
            TensorAlignmentData(
                "sin_out",
                expand("ff_out.pt"),
                expand("torch_out.pt"),
            ),
            TensorAlignmentData(
                "sin_out_grad",
                expand("ff_out_grad.pt"),
                expand("torch_out_grad.pt"),
            ),
        ]
    )
    
def test_transpose():
    out_dir = os.path.join(BASE_DIR, "transpose", "out")
    expand = prepend_dirname_fn(out_dir)
    align_tensors(
        [
            TensorAlignmentData(
                "transpose_out",
                expand("ff_out.pt"),
                expand("torch_out.pt"),
            ),
            TensorAlignmentData(
                "transpose_out_grad",
                expand("ff_out_grad.pt"),
                expand("torch_out_grad.pt"),
            ),
        ]
    )
    

def test_exp():
    out_dir = os.path.join(BASE_DIR, "exp", "out")
    expand = prepend_dirname_fn(out_dir)
    align_tensors(
        [
            TensorAlignmentData(
                "exp_out",
                expand("ff_out.pt"),
                expand("torch_out.pt"),
            ),
            TensorAlignmentData(
                "exp_out_grad",
                expand("ff_out_grad.pt"),
                expand("torch_out_grad.pt"),
            ),
        ]
    )
def test_cos():
    out_dir = os.path.join(BASE_DIR, "cos", "out")
    expand = prepend_dirname_fn(out_dir)
    align_tensors(
        [
            TensorAlignmentData(
                "cos_out",
                expand("ff_out.pt"),
                expand("torch_out.pt"),
            ),
            TensorAlignmentData(
                "cos_out_grad",
                expand("ff_out_grad.pt"),
                expand("torch_out_grad.pt"),
            ),
        ]
    )

def test_scalar_add():
    out_dir = os.path.join(BASE_DIR, "scalar_add", "out")
    expand = prepend_dirname_fn(out_dir)
    align_tensors(
        [
            TensorAlignmentData(
                "scalar_add_out",
                expand("ff_out.pt"),
                expand("torch_out.pt"),
            ),
            TensorAlignmentData(
                "scalar_add_out_grad",
                expand("ff_out_grad.pt"),
                expand("torch_out_grad.pt"),
            ),
        ]
    )
    
def test_scalar_sub():
    out_dir = os.path.join(BASE_DIR, "scalar_sub", "out")
    expand = prepend_dirname_fn(out_dir)
    align_tensors(
        [
            TensorAlignmentData(
                "scalar_sub_out",
                expand("ff_out.pt"),
                expand("torch_out.pt"),
            ),
            TensorAlignmentData(
                "scalar_sub_out_grad",
                expand("ff_out_grad.pt"),
                expand("torch_out_grad.pt"),
            ),
        ]
    )
    
def test_scalar_multiply():
    out_dir = os.path.join(BASE_DIR, "scalar_multiply", "out")
    expand = prepend_dirname_fn(out_dir)
    align_tensors(
        [
            TensorAlignmentData(
                "scalar_multiply_out",
                expand("ff_out.pt"),
                expand("torch_out.pt"),
            ),
            TensorAlignmentData(
                "scalar_multiply_out_grad",
                expand("ff_out_grad.pt"),
                expand("torch_out_grad.pt"),
            ),
        ]
    )
    
def test_scalar_truediv():
    out_dir = os.path.join(BASE_DIR, "scalar_truediv", "out")
    expand = prepend_dirname_fn(out_dir)
    align_tensors(
        [
            TensorAlignmentData(
                "scalar_truediv_out",
                expand("ff_out.pt"),
                expand("torch_out.pt"),
            ),
            TensorAlignmentData(
                "scalar_truediv_out_grad",
                expand("ff_out_grad.pt"),
                expand("torch_out_grad.pt"),
            ),
        ]
    )
    
def test_relu():
    out_dir = os.path.join(BASE_DIR, "relu", "out")
    expand = prepend_dirname_fn(out_dir)
    align_tensors(
        [
            TensorAlignmentData(
                "relu_out",
                expand("ff_out.pt"),
                expand("torch_out.pt"),
            ),
            TensorAlignmentData(
                "relu_out_grad",
                expand("ff_out_grad.pt"),
                expand("torch_out_grad.pt"),
            ),
        ]
    )
    
def test_sigmoid():
    out_dir = os.path.join(BASE_DIR, "sigmoid", "out")
    expand = prepend_dirname_fn(out_dir)
    align_tensors(
        [
            TensorAlignmentData(
                "sigmoid_out",
                expand("ff_out.pt"),
                expand("torch_out.pt"),
            ),
            TensorAlignmentData(
                "sigmoid_out_grad",
                expand("ff_out_grad.pt"),
                expand("torch_out_grad.pt"),
            ),
        ]
    )
    
def test_tanh():
    out_dir = os.path.join(BASE_DIR, "tanh", "out")
    expand = prepend_dirname_fn(out_dir)
    align_tensors(
        [
            TensorAlignmentData(
                "tanh_out",
                expand("ff_out.pt"),
                expand("torch_out.pt"),
            ),
            TensorAlignmentData(
                "tanh_out_grad",
                expand("ff_out_grad.pt"),
                expand("torch_out_grad.pt"),
            ),
        ]
    )
    
def test_identity():
    out_dir = os.path.join(BASE_DIR, "identity", "out")
    expand = prepend_dirname_fn(out_dir)
    align_tensors(
        [
            TensorAlignmentData(
                "identity_out",
                expand("ff_out.pt"),
                expand("torch_out.pt"),
            ),
            TensorAlignmentData(
                "identity_out_grad",
                expand("ff_out_grad.pt"),
                expand("torch_out_grad.pt"),
            ),
        ]
    )
    