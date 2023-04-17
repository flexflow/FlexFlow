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

def test_linear():
    out_dir = os.path.join(BASE_DIR, "linear", "out")
    expand = prepend_dirname_fn(out_dir)
    align_tensors(
        [
            TensorAlignmentData(
                "linear_out",
                expand("ff_out.pt"),
                expand("torch_out.pt"),
            ),
            TensorAlignmentData(
                "linear_out_grad",
                expand("ff_out_grad.pt"),
                expand("torch_out_grad.pt"),
            ),
            TensorAlignmentData(
                "linear_weight_grad",
                expand("ff_weight_grad.pt"),
                expand("torch_weight_grad.pt"),
            ),
            TensorAlignmentData(
                "linear_bias_grad",
                expand("ff_bias_grad.pt"),
                expand("torch_bias_grad.pt")
            )
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

def test_subtract():
    _test_operator('subtract')


def test_multiply():
    _test_operator('multiply')


def test_pool2d():
    _test_operator('pool2d')


def test_reducesum():
    _test_operator('reducesum')


def test_reshape():
    _test_operator('reshape')


def test_flat():
    _test_operator('flat')


def test_sin():
    _test_operator('sin')


def test_transpose():
    _test_operator('transpose')


def test_exp():
    _test_operator('exp')


def test_cos():
    _test_operator('cos')


def test_scalar_add():
    _test_operator('scalar_add')


def test_scalar_sub():
    _test_operator('scalar_sub')


def test_scalar_multiply():
    _test_operator('scalar_multiply')


def test_scalar_truediv():
    _test_operator('scalar_truediv')


def test_relu():
    _test_operator('relu')


def test_sigmoid():
    _test_operator('sigmoid')


def test_tanh():
    _test_operator('tanh')


def test_identity():
    _test_operator('identity')
    
    
def test_linear():
    _test_operator('linear')
    
    
# def test_max():
#     _test_operator('max')
    
    
# def test_min():
#     _test_operator('min')
    
def test_gather():
    _test_operator('gather')


def _test_operator(operater_name):
    out_dir = os.path.join(BASE_DIR, operater_name)
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