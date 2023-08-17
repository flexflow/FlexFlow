from align_utils import TensorAlignmentData, align_tensors
import os
import sys
from typing import Callable

sys.path.append("./align/")


BASE_DIR = "tests/align/out"
param_weight_op = {'conv2d', 'embedding', 'view_embedding', 'linear'}
param_bias_op = {'conv2d', 'linear'}
no_grad_op = {"getitem"}


def prepend_dirname_fn(dirname: str) -> Callable[[str], str]:
    def f(filename):
        return os.path.join(dirname, filename)
    return f


def test_embedding():
    _test_operator('embedding')


def test_view_embedding():
    _test_operator('view_embedding')


def test_getitem():
    _test_operator('getitem')


def test_conv2d():
    _test_operator('conv2d')


def test_add():
    _test_operator('add')


def test_concat():
    _test_operator('concat')


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

    if (operater_name in no_grad_op):
        align_tensors(
            [
                TensorAlignmentData(
                    operater_name + "_out",
                    expand("ff_out.pt"),
                    expand("torch_out.pt"),
                ),
            ]
        )
        return

    # test output
    align_tensors(
        [
            TensorAlignmentData(
                operater_name + "_out",
                expand("ff_out.pt"),
                expand("torch_out.pt"),
            ),
            TensorAlignmentData(
                operater_name + "_out_grad",
                expand("ff_out_grad.pt"),
                expand("torch_out_grad.pt"),
            ),
        ]
    )

    # test weight
    if (operater_name in param_weight_op):
        align_tensors(
            [
                TensorAlignmentData(
                    operater_name + "_weight_grad",
                    expand("ff_weight_grad.pt"),
                    expand("torch_weight_grad.pt"),
                ),
            ]
        )
    # test bias
    if (operater_name in param_bias_op):
        align_tensors(
            [
                TensorAlignmentData(
                    operater_name + "_bias_grad",
                    expand("ff_bias_grad.pt"),
                    expand("torch_bias_grad.pt")
                )
            ]
        )
