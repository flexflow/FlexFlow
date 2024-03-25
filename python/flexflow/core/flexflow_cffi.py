# Copyright 2023 CMU, Facebook, LANL, MIT, NVIDIA, and Stanford (alphabetical)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from __future__ import absolute_import, division, print_function, unicode_literals

import warnings
import numpy as np
from .flexflow_logger import fflogger
from flexflow.type import (
    ActiMode,
    RegularizerMode,
    AggrMode,
    PoolType,
    DataType,
    LossType,
    CompMode,
    MetricsType,
    InferenceMode,
    RequestType,
    ModelType,
    OpType,
    ParameterSyncType,
    enum_to_int,
    int_to_enum,
)
from flexflow.config import *
from .flexflowlib import ffi, flexflow_library
from typing import Union, List

def ffc():
    if not flexflow_already_initialized():
        raise RuntimeError("Cannot use FlexFlow library before initializing FlexFlow")
    ffc = flexflow_library.lib
    if ffc is None:
        raise RuntimeError("FlexFlow library is None")
    return ffc


ff_tracing_id = 200

warnings.simplefilter("always", DeprecationWarning)


def get_c_name(name):
    if name is None:
        return ffi.NULL
    else:
        return ffi.new("char[]", name.encode("utf-8"))


def get_datatype_size(datatype):
    if datatype == DataType.DT_HALF:
        return 2
    if datatype == DataType.DT_FLOAT:
        return 4
    elif datatype == DataType.DT_DOUBLE:
        return 8
    elif datatype == DataType.DT_INT32:
        return 4
    elif datatype == DataType.DT_INT64:
        return 8
    else:
        assert 0, "unknow datatype" + str(datatype)
        return 0


# -----------------------------------------------------------------------
# Op
# -----------------------------------------------------------------------
class Op(object):
    __slots__ = ["handle", "idx", "name"]

    def __init__(self, handle, idx=None, name=None):
        assert ffi.typeof(handle) == ffi.typeof("flexflow_op_t"), "Op handle is wrong"
        self.handle = handle
        self.idx = idx
        self.name = name

    def get_number_parameters(self):
        return ffc().flexflow_op_get_num_parameters(self.handle)

    def get_parameter_by_id(self, id):
        handle = ffc().flexflow_op_get_parameter_by_id(self.handle, id)
        return Parameter(handle)

    def get_number_inputs(self):
        return ffc().flexflow_op_get_num_inputs(self.handle)

    def get_input_by_id(self, id):
        handle = ffc().flexflow_op_get_input_by_id(self.handle, id)
        return Tensor(handle, False)

    def get_number_outputs(self):
        return ffc().flexflow_op_get_num_outputs(self.handle)

    def get_output_by_id(self, id):
        handle = ffc().flexflow_op_get_output_by_id(self.handle, id)
        return Tensor(handle, False)

    def init(self, model):
        ffc().flexflow_op_init(self.handle, model.handle)

    def forward(self, model):
        ffc().flexflow_op_forward(self.handle, model.handle)
        # return Tensor(handle)

    def _add_to_model(self, model):
        ffc().flexflow_op_add_to_model(self.handle, model.handle)

    def get_output_tensor(self):
        return self.get_output_by_id(0)


# -----------------------------------------------------------------------
# Exp
# -----------------------------------------------------------------------
class Exp(Op):
    def __init__(self, handle, idx=None, name=None):
        super(Exp, self).__init__(handle, idx, name)


# -----------------------------------------------------------------------
# Sin
# -----------------------------------------------------------------------
class Sin(Op):
    def __init__(self, handle, idx=None, name=None):
        super(Sin, self).__init__(handle, idx, name)


# -----------------------------------------------------------------------
# Cos
# -----------------------------------------------------------------------
class Cos(Op):
    def __init__(self, handle, idx=None, name=None):
        super(Cos, self).__init__(handle, idx, name)


# -----------------------------------------------------------------------
# Add
# -----------------------------------------------------------------------
class Add(Op):
    def __init__(self, handle, idx=None, name=None):
        super(Add, self).__init__(handle, idx, name)


# -----------------------------------------------------------------------
# Subtract
# -----------------------------------------------------------------------
class Subtract(Op):
    def __init__(self, handle, idx=None, name=None):
        super(Subtract, self).__init__(handle, idx, name)


# -----------------------------------------------------------------------
# Multiply
# -----------------------------------------------------------------------
class Multiply(Op):
    def __init__(self, handle, idx=None, name=None):
        super(Multiply, self).__init__(handle, idx, name)


# -----------------------------------------------------------------------
# Divide
# -----------------------------------------------------------------------
class Divide(Op):
    def __init__(self, handle, idx=None, name=None):
        super(Divide, self).__init__(handle, idx, name)


# -----------------------------------------------------------------------
# Max
# -----------------------------------------------------------------------
class Max(Op):
    def __init__(self, handle, idx=None, name=None):
        super(Max, self).__init__(handle, idx, name)


# -----------------------------------------------------------------------
# Min
# -----------------------------------------------------------------------
class Min(Op):
    def __init__(self, handle, idx=None, name=None):
        super(Min, self).__init__(handle, idx, name)


# -----------------------------------------------------------------------
# ReduceSum
# -----------------------------------------------------------------------
class ReduceSum(Op):
    def __init__(self, handle, idx=None, name=None):
        super(ReduceSum, self).__init__(handle, idx, name)


# -----------------------------------------------------------------------
# Conv2D
# -----------------------------------------------------------------------
class Conv2D(Op):
    def __init__(self, handle, idx=None, name=None):
        super(Conv2D, self).__init__(handle, idx, name)

    def get_weight_tensor(self):
        return self.get_parameter_by_id(0)

    def get_bias_tensor(self):
        return self.get_parameter_by_id(1)

    def get_input_tensor(self):
        return self.get_input_by_id(0)

    def get_output_tensor(self):
        return self.get_output_by_id(0)


# -----------------------------------------------------------------------
# Pool2D
# -----------------------------------------------------------------------
class Pool2D(Op):
    def __init__(self, handle, idx=None, name=None):
        super(Pool2D, self).__init__(handle, idx, name)

    def get_input_tensor(self):
        return self.get_input_by_id(0)

    def get_output_tensor(self):
        return self.get_output_by_id(0)


# -----------------------------------------------------------------------
# Linear
# -----------------------------------------------------------------------
class Linear(Op):
    def __init__(self, handle, idx=None, name=None):
        super(Linear, self).__init__(handle, idx, name)

    def get_weight_tensor(self):
        return self.get_parameter_by_id(0)

    def get_bias_tensor(self):
        return self.get_parameter_by_id(1)

    def get_input_tensor(self):
        return self.get_input_by_id(0)

    def get_output_tensor(self):
        return self.get_output_by_id(0)


# -----------------------------------------------------------------------
# Flat
# -----------------------------------------------------------------------
class Flat(Op):
    def __init__(self, handle, idx=None, name=None):
        super(Flat, self).__init__(handle, idx, name)

    def get_input_tensor(self):
        return self.get_input_by_id(0)

    def get_output_tensor(self):
        return self.get_output_by_id(0)


# -----------------------------------------------------------------------
# Softmax
# -----------------------------------------------------------------------
class Softmax(Op):
    def __init__(self, handle, idx=None, name=None):
        super(Softmax, self).__init__(handle, idx, name)


# -----------------------------------------------------------------------
# Embedding
# -----------------------------------------------------------------------
class Embedding(Op):
    def __init__(self, handle, idx=None, name=None):
        super(Embedding, self).__init__(handle, idx, name)

    def get_weight_tensor(self):
        return self.get_parameter_by_id(0)


# -----------------------------------------------------------------------
# Concat
# -----------------------------------------------------------------------
class Concat(Op):
    def __init__(self, handle, idx=None, name=None):
        super(Concat, self).__init__(handle, idx, name)


# -----------------------------------------------------------------------
# BatchNorm
# -----------------------------------------------------------------------
class BatchNorm(Op):
    def __init__(self, handle, idx=None, name=None):
        super(BatchNorm, self).__init__(handle, idx, name)


# -----------------------------------------------------------------------
# LayerNorm
# -----------------------------------------------------------------------
class LayerNorm(Op):
    def __init__(self, handle, idx=None, name=None):
        super(LayerNorm, self).__init__(handle, idx, name)

    def get_weight_tensor(self):
        return self.get_parameter_by_id(0)

    def get_bias_tensor(self):
        return self.get_parameter_by_id(1)


# -----------------------------------------------------------------------
# ResidualLayerNorm
# -----------------------------------------------------------------------
class ResidualLayerNorm(Op):
    def __init__(self, handle, idx=None, name=None):
        super(ResidualLayerNorm, self).__init__(handle, idx, name)

    def get_weight_tensor(self):
        return self.get_parameter_by_id(1)

    def get_bias_tensor(self):
        return self.get_parameter_by_id(2)


# -----------------------------------------------------------------------
# AddBiasResidualLayerNorm
# -----------------------------------------------------------------------
class AddBiasResidualLayerNorm(Op):
    def __init__(self, handle, idx=None, name=None):
        super(AddBiasResidualLayerNorm, self).__init__(handle, idx, name)

    def get_attn_bias_tensor(self):
        return self.get_parameter_by_id(0)

    def get_weight_tensor(self):
        return self.get_parameter_by_id(1)

    def get_bias_tensor(self):
        return self.get_parameter_by_id(2)


# -----------------------------------------------------------------------
# SigmoidSiluMulti
# -----------------------------------------------------------------------
class SigmoidSiluMulti(Op):
    def __init__(self, handle, idx=None, name=None):
        super(SigmoidSiluMulti, self).__init__(handle, idx, name)


# -----------------------------------------------------------------------
# Dropout
# -----------------------------------------------------------------------
class Dropout(Op):
    def __init__(self, handle, idx=None, name=None):
        super(Dropout, self).__init__(handle, idx, name)


# -----------------------------------------------------------------------
# ScalarMultiply
# -----------------------------------------------------------------------
class ScalarMultiply(Op):
    def __init__(self, handle, idx=None, name=None):
        super(ScalarMultiply, self).__init__(handle, idx, name)


# -----------------------------------------------------------------------
# ScalarAdd
# -----------------------------------------------------------------------
class ScalarAdd(Op):
    def __init__(self, handle, idx=None, name=None):
        super(ScalarAdd, self).__init__(handle, idx, name)


# -----------------------------------------------------------------------
# ScalarSub
# -----------------------------------------------------------------------
class ScalarSub(Op):
    def __init__(self, handle, idx=None, name=None):
        super(ScalarSub, self).__init__(handle, idx, name)


# -----------------------------------------------------------------------
# ScalarTrueDiv
# -----------------------------------------------------------------------
class ScalarTrueDiv(Op):
    def __init__(self, handle, idx=None, name=None):
        super(ScalarTrueDiv, self).__init__(handle, idx, name)


# -----------------------------------------------------------------------
# Rsqrt
# -----------------------------------------------------------------------
class Rsqrt(Op):
    def __init__(self, handle, idx=None, name=None):
        super(Rsqrt, self).__init__(handle, idx, name)


# -----------------------------------------------------------------------
# Pow
# -----------------------------------------------------------------------
class Pow(Op):
    def __init__(self, handle, idx=None, name=None):
        super(Pow, self).__init__(handle, idx, name)


# -----------------------------------------------------------------------
# Mean
# -----------------------------------------------------------------------
class Mean(Op):
    def __init__(self, handle, idx=None, name=None):
        super(Mean, self).__init__(handle, idx, name)


# -----------------------------------------------------------------------
# Relu
# -----------------------------------------------------------------------
class Relu(Op):
    def __init__(self, handle, idx=None, name=None):
        super(Relu, self).__init__(handle, idx, name)


# -----------------------------------------------------------------------
# Gelu
# -----------------------------------------------------------------------
class Gelu(Op):
    def __init__(self, handle, idx=None, name=None):
        super(Gelu, self).__init__(handle, idx, name)


# -----------------------------------------------------------------------
# Sigmod
# -----------------------------------------------------------------------
class Sigmoid(Op):
    def __init__(self, handle, idx=None, name=None):
        super(Sigmoid, self).__init__(handle, idx, name)


# -----------------------------------------------------------------------
# Tanh
# -----------------------------------------------------------------------
class Tanh(Op):
    def __init__(self, handle, idx=None, name=None):
        super(Tanh, self).__init__(handle, idx, name)


# -----------------------------------------------------------------------
# Elu
# -----------------------------------------------------------------------
class Elu(Op):
    def __init__(self, handle, idx=None, name=None):
        super(Elu, self).__init__(handle, idx, name)


# -----------------------------------------------------------------------
# Batch_Norm
# -----------------------------------------------------------------------
class Batch_Norm(Op):
    def __init__(self, handle, idx=None, name=None):
        super(Batch_Norm, self).__init__(handle, idx, name)


# -----------------------------------------------------------------------
# Batch_Matmul
# -----------------------------------------------------------------------
class Batch_Matmul(Op):
    def __init__(self, handle, idx=None, name=None):
        super(Batch_Matmul, self).__init__(handle, idx, name)


# -----------------------------------------------------------------------
# Split
# -----------------------------------------------------------------------
class Split(Op):
    def __init__(self, handle, idx=None, name=None):
        super(Split, self).__init__(handle, idx, name)


# -----------------------------------------------------------------------
# Reshape
# -----------------------------------------------------------------------
class Reshape(Op):
    def __init__(self, handle, idx=None, name=None):
        super(Reshape, self).__init__(handle, idx, name)


# -----------------------------------------------------------------------
# Gather
# -----------------------------------------------------------------------
class Gather(Op):
    def __init__(self, handle, idx=None, name=None):
        super(Gather, self).__init__(handle, idx, name)


# -----------------------------------------------------------------------
# Identity
# -----------------------------------------------------------------------
class Identity(Op):
    def __init__(self, handle, idx=None, name=None):
        super(Identity, self).__init__(handle, idx, name)


# -----------------------------------------------------------------------
# Transpose
# -----------------------------------------------------------------------
class Transpose(Op):
    def __init__(self, handle, idx=None, name=None):
        super(Transpose, self).__init__(handle, idx, name)


# -----------------------------------------------------------------------
# Reverse
# -----------------------------------------------------------------------
class Reverse(Op):
    def __init__(self, handle, idx=None, name=None):
        super(Reverse, self).__init__(handle, idx, name)


# -----------------------------------------------------------------------
# MultiHeadAttention
# -----------------------------------------------------------------------
class MultiHeadAttention(Op):
    def __init__(self, handle, idx=None, name=None):
        super(MultiHeadAttention, self).__init__(handle, idx, name)


# -----------------------------------------------------------------------
# Incremental MultiHeadAttention
# -----------------------------------------------------------------------
class IncMultiHeadAttention(Op):
    def __init__(self, handle, idx=None, name=None):
        super(IncMultiHeadAttention, self).__init__(handle, idx, name)


# -----------------------------------------------------------------------
# Speculative Incremental MultiHeadAttention
# -----------------------------------------------------------------------
class SpecIncMultiHeadSelfAttention(Op):
    def __init__(self, handle, idx=None, name=None):
        super(SpecIncMultiHeadSelfAttention, self).__init__(handle, idx, name)


# -----------------------------------------------------------------------
# TreeVerify Incremental MultiHeadAttention
# -----------------------------------------------------------------------
class TreeIncMultiHeadSelfAttention(Op):
    def __init__(self, handle, idx=None, name=None):
        super(TreeIncMultiHeadSelfAttention, self).__init__(handle, idx, name)


# -----------------------------------------------------------------------
# RMS Norm
# -----------------------------------------------------------------------
class RMSNorm(Op):
    def __init__(self, handle, idx=None, name=None):
        super(RMSNorm, self).__init__(handle, idx, name)


# -----------------------------------------------------------------------
# Residual RMS Norm
# -----------------------------------------------------------------------
class ResidualRMSNorm(Op):
    def __init__(self, handle, idx=None, name=None):
        super(ResidualRMSNorm, self).__init__(handle, idx, name)


# -----------------------------------------------------------------------
# ArgTopK
# -----------------------------------------------------------------------
class ArgTopK(Op):
    def __init__(self, handle, idx=None, name=None):
        super(ArgTopK, self).__init__(handle, idx, name)


# -----------------------------------------------------------------------
# BeamTopK
# -----------------------------------------------------------------------
class BeamTopK(Op):
    def __init__(self, handle, idx=None, name=None):
        super(BeamTopK, self).__init__(handle, idx, name)


# -----------------------------------------------------------------------
# Sampling
# -----------------------------------------------------------------------
class Sampling(Op):
    def __init__(self, handle, idx=None, name=None):
        super(Sampling, self).__init__(handle, idx, name)


# -----------------------------------------------------------------------
# ArgMax
# -----------------------------------------------------------------------
class ArgMax(Op):
    def __init__(self, handle, idx=None, name=None):
        super(ArgMax, self).__init__(handle, idx, name)


# -----------------------------------------------------------------------
# flexflow_op_t handle to Op
# -----------------------------------------------------------------------
def convert_op_handle_to_op(op_type, handle, idx=None, name=None):
    if op_type == OpType.CONV2D:
        return Conv2D(handle, idx, name)
    elif op_type == OpType.POOL2D:
        return Pool2D(handle, idx, name)
    elif op_type == OpType.LINEAR:
        return Linear(handle, idx, name)
    elif op_type == OpType.EMBEDDING:
        return Embedding(handle, idx, name)
    elif op_type == OpType.FLAT:
        return Flat(handle, idx, name)
    elif op_type == OpType.CONCAT:
        return Concat(handle, idx, name)
    elif op_type == OpType.SOFTMAX:
        return Softmax(handle, idx, name)
    elif op_type == OpType.EXP:
        return Exp(handle, idx, name)
    elif op_type == OpType.SIN:
        return Sin(handle, idx, name)
    elif op_type == OpType.COS:
        return Cos(handle, idx, name)
    elif op_type == OpType.ADD:
        return Add(handle, idx, name)
    elif op_type == OpType.SUBTRACT:
        return Subtract(handle, idx, name)
    elif op_type == OpType.MULTIPLY:
        return Multiply(handle, idx, name)
    elif op_type == OpType.DIVIDE:
        return Divide(handle, idx, name)
    elif op_type == OpType.MAX:
        return Max(handle, idx, name)
    elif op_type == OpType.MIN:
        return Min(handle, idx, name)
    elif op_type == OpType.REDUCE_SUM:
        return ReduceSum(handle, idx, name)
    elif op_type == OpType.MSELOSS:
        return MSELoss(handle, idx, name)
    elif op_type == OpType.SCALAR_MULTIPLY:
        return ScalarMultiply(handle, idx, name)
    elif op_type == OpType.SCALAR_ADD:
        return ScalarAdd(handle, idx, name)
    elif op_type == OpType.SCALAR_SUB:
        return ScalarSub(handle, idx, name)
    elif op_type == OpType.SCALAR_FLOORDIV:
        return ScalarFloorDiv(handle, idx, name)
    elif op_type == OpType.SCALAR_TRUEDIV:
        return ScalarTrueDiv(handle, idx, name)
    elif op_type == OpType.GELU:
        return Gelu(handle, idx, name)
    elif op_type == OpType.RELU:
        return Relu(handle, idx, name)
    elif op_type == OpType.SIGMOID:
        return Sigmoid(handle, idx, name)
    elif op_type == OpType.TANH:
        return Tanh(handle, idx, name)
    elif op_type == OpType.ELU:
        return Elu(handle, idx, name)
    elif op_type == OpType.DROPOUT:
        return Dropout(handle, idx, name)
    elif op_type == OpType.BATCH_NORM:
        return BatchNorm(handle, idx, name)
    elif op_type == OpType.LAYER_NORM:
        return LayerNorm(handle, idx, name)
    elif op_type == OpType.RESIDUAL_LAYERNORM:
        return ResidualLayerNorm(handle, idx, name)
    elif op_type == OpType.ADD_BIAS_RESIDUAL_LAYERNORM:
        return AddBiasResidualLayerNorm(handle, idx, name)
    elif op_type == OpType.SIGMOID_SILU_MULTI:
        return SigmoidSiluMulti(handle, idx, name)
    elif op_type == OpType.BATCH_MATMUL:
        return Batch_Matmul(handle, idx, name)
    elif op_type == OpType.SPLIT:
        return Split(handle, idx, name)
    elif op_type == OpType.RESHAPE:
        return Reshape(handle, idx, name)
    elif op_type == OpType.IDENTITY:
        return Identity(handle, idx, name)
    elif op_type == OpType.TRANSPOSE:
        return Transpose(handle, idx, name)
    elif op_type == OpType.REVERSE:
        return Reverse(handle, idx, name)
    elif op_type == OpType.MULTIHEAD_ATTENTION:
        return MultiHeadAttention(handle, idx, name)
    elif op_type == OpType.INC_MULTIHEAD_ATTENTION:
        return IncMultiHeadAttention(handle, idx, name)
    elif op_type == OpType.SPEC_INC_MULTIHEAD_SELF_ATTENTION:
        return SpecIncMultiHeadSelfAttention(handle, idx, name)
    elif op_type == OpType.TREE_INC_MULTIHEAD_SELF_ATTENTION:
        return TreeIncMultiHeadSelfAttention(handle, idx, name)
    elif op_type == OpType.RMS_NORM:
        return RMSNorm(handle, idx, name)
    elif op_type == OpType.RESIDUAL_RMS_NORM:
        return ResidualRMSNorm(handle, idx, name)
    elif op_type == OpType.ARG_TOPK:
        return ArgTopK(handle, idx, name)
    elif op_type == OpType.BEAM_TOPK:
        return BeamTopK(handle, idx, name)
    elif op_type == OpType.SAMPLING:
        return Sampling(handle, idx, name)
    elif op_type == OpType.ARGMAX:
        return ArgMax(handle, idx, name)
    elif op_type == OpType.RSQRT:
        return Rsqrt(handle, idx, name)
    elif op_type == OpType.POW:
        return Pow(handle, idx, name)
    elif op_type == OpType.MEAN:
        return Mean(handle, idx, name)
    elif op_type == OpType.GATHER:
        return Gather(handle, idx, name)
    else:
        assert 0, "unknown layer type {}".format(op_type)
        return None


# -----------------------------------------------------------------------
# FFConfig
# -----------------------------------------------------------------------


class FFConfig(object):
    __slots__ = ["handle", "_handle", "enable_tracing"]

    def __init__(self):
        self.handle = ffc().flexflow_config_create()
        self._handle = ffi.gc(self.handle, ffc().flexflow_config_destroy)
        self.enable_tracing = False

    def parse_args(self):
        ffc().flexflow_config_parse_args_default(self.handle)

    @property
    def batch_size(self):
        return ffc().flexflow_config_get_batch_size(self.handle)

    @property
    def workers_per_node(self):
        return ffc().flexflow_config_get_workers_per_node(self.handle)

    @property
    def num_nodes(self):
        return ffc().flexflow_config_get_num_nodes(self.handle)

    @property
    def epochs(self):
        return ffc().flexflow_config_get_epochs(self.handle)

    @property
    def enable_control_replication(self):
        return ffc().flexflow_config_get_enable_control_replication(self.handle)

    @property
    def data_parallelism_degree(self):
        return ffc().flexflow_config_get_data_parallelism_degree(self.handle)

    @data_parallelism_degree.setter
    def data_parallelism_degree(self, value):
        if type(value) is not int:
            raise ValueError(
                "The data parallelism degree must be specified as an integer number"
            )
        elif value < 1:
            raise ValueError("The data parallelism degree cannot be lower than 1")
        ffc().flexflow_config_set_data_parallelism_degree(self.handle, value)

    @property
    def tensor_parallelism_degree(self):
        return ffc().flexflow_config_get_tensor_parallelism_degree(self.handle)

    @tensor_parallelism_degree.setter
    def tensor_parallelism_degree(self, value):
        if type(value) is not int:
            raise ValueError(
                "The tensor parallelism degree must be specified as an integer number"
            )
        elif value < 1:
            raise ValueError("The tensor parallelism degree cannot be lower than 1")
        ffc().flexflow_config_set_tensor_parallelism_degree(self.handle, value)

    @property
    def pipeline_parallelism_degree(self):
        return ffc().flexflow_config_get_pipeline_parallelism_degree(self.handle)

    @pipeline_parallelism_degree.setter
    def pipeline_parallelism_degree(self, value):
        if type(value) is not int:
            raise ValueError(
                "The pipeline parallelism degree must be specified as an integer number"
            )
        elif value < 1:
            raise ValueError("The pipeline parallelism degree cannot be lower than 1")
        ffc().flexflow_config_set_pipeline_parallelism_degree(self.handle, value)

    @property
    def python_data_loader_type(self):
        return ffc().flexflow_config_get_python_data_loader_type(self.handle)

    @property
    def cpu_offload(self):
        return ffc().flexflow_config_get_offload(self.handle)

    def get_current_time(self):
        return ffc().flexflow_get_current_time(self.handle)

    def begin_trace(self, trace_id):
        if self.enable_tracing:
            ffc().flexflow_begin_trace(self.handle, trace_id)

    def end_trace(self, trace_id):
        if self.enable_tracing:
            ffc().flexflow_end_trace(self.handle, trace_id)


# -----------------------------------------------------------------------
# Tensor
# -----------------------------------------------------------------------


class Tensor(object):
    __slots__ = [
        "p_handle",
        "handle",
        "_handle",
        "num_dims",
        "dims",
        "data_type",
        "owner_op",
        "mapped",
    ]

    def __init__(self, handle, deallocate=True, owner_op_type=None, p_handle=None):
        if handle == None and ffi.typeof(p_handle) == ffi.typeof("flexflow_tensor_t*"):
            self.p_handle = p_handle
            self.handle = self.p_handle[0]
        elif handle != None and ffi.typeof(handle) == ffi.typeof("flexflow_tensor_t"):
            self.p_handle = 0
            self.handle = handle
        # elif handle != None and ffi.typeof(handle) == ffi.typeof('flexflow_tensor_t'):
        #  self.p_handle = ffi.new('flexflow_tensor_t *')
        #  self.p_handle.impl = handle.impl
        #  self.handle = self.p_handle[0]
        else:
            assert 0, "Tensor handle is wrong"
        self.num_dims = 0
        self.dims = 0
        self.mapped = False
        self.__get_dims()
        self.__get_data_type()
        # if (deallocate == True):
        #   self._handle = ffi.gc(self.handle, ffc().flexflow_tensor_destroy)
        # if (self.is_mapped() == True):
        #   self.mapped = True

        if owner_op_type != None:
            self.__get_owner_op(owner_op_type)
            assert self.owner_op != None

    def inline_map(self, ffmodel, ffconfig):
        assert self.mapped == False, "Tensor is already mapped."
        ffc().flexflow_tensor_inline_map(self.handle, ffmodel.handle, ffconfig.handle)
        self.mapped = True
        assert self.num_dims > 0, "check dims"

    def inline_unmap(self, ffmodel, ffconfig):
        assert self.mapped == True, "Tensor is not inline mapped."
        ffc().flexflow_tensor_inline_unmap(self.handle, ffmodel.handle, ffconfig.handle)
        self.mapped = False

    def get_array(self, ffmodel, ffconfig):
        assert self.mapped == True, "Tensor is not mapped."
        raw_ptr = self.__get_raw_ptr(ffmodel, ffconfig, self.data_type)
        raw_ptr_int = int(ffi.cast("uintptr_t", raw_ptr))
        fflogger.debug("raw_ptr: %s, %d" % (str(raw_ptr), raw_ptr_int))
        strides = None
        if self.num_dims >= 1 or self.num_dims <= 4:
            shape = self.dims
        else:
            assert 0, "unknow num_dims"
        initializer = RegionNdarray(shape, self.data_type, raw_ptr_int, strides, False)
        array = np.asarray(initializer)
        # print("stride", array.__array_interface__['strides'])
        return array

    def get_flat_array(self, ffmodel, ffconfig):
        assert self.mapped == True, "Tensor is not mapped."
        raw_ptr = self.__get_raw_ptr(ffmodel, ffconfig, self.data_type)
        raw_ptr_int = int(ffi.cast("uintptr_t", raw_ptr))
        fflogger.debug("raw_ptr: %s, %d" % (str(raw_ptr), raw_ptr_int))
        strides = None
        if self.num_dims >= 1 or self.num_dims <= 4:
            shape_prod = np.prod(self.dims)
            shape = (shape_prod,)
        else:
            assert 0, "unknown num_dims"
        initializer = RegionNdarray(shape, self.data_type, raw_ptr_int, strides, False)
        array = np.asarray(initializer)
        return array

    def attach_numpy_array(self, ffmodel, ffconfig, np_array):
        assert (
            np_array.__array_interface__["strides"] == None
        ), "numpy array strides is not None"
        np_shape = np_array.shape
        num_dims = len(np_shape)
        assert num_dims == self.num_dims, "please check dims (%d == %d)" % (
            num_dims,
            self.num_dims,
        )
        for i in range(0, num_dims):
            assert (
                np_shape[i] == self.dims[i]
            ), "please check shape dim %d (%d == %d)" % (i, np_shape[i], self.dims[i])
        np_raw_ptr = np_array.__array_interface__["data"]
        raw_ptr = ffi.cast("void*", np_raw_ptr[0])
        fflogger.debug(
            "attach numpy array: %s, %s, %s"
            % (str(np_raw_ptr), str(raw_ptr), hex(np_raw_ptr[0]))
        )
        self.__attach_raw_ptr(ffmodel, ffconfig, raw_ptr)

    def detach_numpy_array(self, ffconfig):
        self.__detach_raw_ptr(ffconfig)

    def is_mapped(self):
        return ffc().flexflow_tensor_is_mapped(self.handle)

    def set_tensor(self, ffmodel, np_array):
        assert (
            np_array.__array_interface__["strides"] == None
        ), "Parameter set_weights, numpy array strides is not None"
        np_shape = np_array.shape
        num_dims = len(np_shape)
        assert num_dims == self.num_dims, "please check dims (%d == %d)" % (
            num_dims,
            self.num_dims,
        )
        for i in range(0, num_dims):
            assert (
                np_shape[i] == self.dims[i]
            ), "please check shape dim %d (%d == %d)" % (i, np_shape[i], self.dims[i])
        c_dims = ffi.new("int[]", self.dims)
        np_raw_ptr = np_array.__array_interface__["data"]
        if np_array.dtype == np.float16:
            assert self.data_type == DataType.DT_HALF, "Wrong datatype"
            raw_ptr = ffi.cast("half*", np_raw_ptr[0])
            ret_val = ffc().flexflow_tensor_set_tensor_float(
                self.handle, ffmodel.handle, num_dims, c_dims, raw_ptr
            )
        elif np_array.dtype == np.float32:
            assert self.data_type == DataType.DT_FLOAT, "Wrong datatype"
            raw_ptr = ffi.cast("float*", np_raw_ptr[0])
            ret_val = ffc().flexflow_tensor_set_tensor_float(
                self.handle, ffmodel.handle, num_dims, c_dims, raw_ptr
            )
        elif np_array.dtype == np.int32:
            assert self.data_type == DataType.DT_INT32, "Wrong datatype"
            raw_ptr = ffi.cast("int*", np_raw_ptr[0])
            ret_val = ffc().flexflow_tensor_set_tensor_int(
                self.handle, ffmodel.handle, num_dims, c_dims, raw_ptr
            )
        else:
            assert 0, "Unsupported datatype"
        fflogger.debug(
            "set tensor raw_ptr: %s, %s, %s, %s"
            % (str(raw_ptr), str(np_raw_ptr[0]), hex(np_raw_ptr[0]), str(np_shape))
        )
        assert ret_val == True, ret_val

    def get_tensor(self, ffmodel):
        shape = self.dims
        if self.data_type == DataType.DT_HALF:
            np_array = np.empty(shape, dtype=np.float16)
        elif self.data_type == DataType.DT_FLOAT:
            np_array = np.empty(shape, dtype=np.float32)
        elif self.data_type == DataType.DT_INT32:
            np_array = np.empty(shape, dtype=np.int32)
        elif self.data_type == DataType.DT_INT64:
            np_array = np.empty(shape, dtype=np.int64)
        else:
            assert 0, f"Unsupported datatype: {self.data_type}"
        np_raw_ptr = np_array.__array_interface__["data"]
        if np_array.dtype == np.float32:
            raw_ptr = ffi.cast("float*", np_raw_ptr[0])
            ret_val = ffc().flexflow_tensor_get_tensor_float(
                self.handle, ffmodel.handle, raw_ptr, False
            )
        elif np_array.dtype == np.int32:
            raw_ptr = ffi.cast("int*", np_raw_ptr[0])
            ret_val = ffc().flexflow_tensor_get_tensor_int(
                self.handle, ffmodel.handle, raw_ptr, False
            )
        elif np_array.dtype == np.int64:
            raw_ptr = ffi.cast("int64_t*", np_raw_ptr[0])
            ret_val = ffc().flexflow_tensor_get_tensor_int64(
                self.handle, ffmodel.handle, raw_ptr, False
            )
        fflogger.debug(
            "get weights raw_ptr: %s, %s, %s, %s"
            % (str(raw_ptr), str(np_raw_ptr[0]), hex(np_raw_ptr[0]), str(shape))
        )
        assert ret_val == True
        return np_array

    def get_gradients(self, ffmodel, comm_type):
        shape = self.dims
        if self.data_type == DataType.DT_HALF:
            np_array = np.empty(shape, dtype=np.float16)
        elif self.data_type == DataType.DT_FLOAT:
            np_array = np.empty(shape, dtype=np.float32)
        elif self.data_type == DataType.DT_INT32:
            np_array = np.empty(shape, dtype=np.int32)
        elif self.data_type == DataType.DT_INT64:
            np_array = np.empty(shape, dtype=np.int64)
        else:
            assert 0, f"Unsupported datatype: {self.data_type}"
        np_raw_ptr = np_array.__array_interface__["data"]
        c_comm_type = enum_to_int(ParameterSyncType, comm_type)
        if np_array.dtype == np.float32:
            raw_ptr = ffi.cast("float*", np_raw_ptr[0])
            ret_val = ffc().flexflow_tensor_get_tensor_float(
                self.handle, ffmodel.handle, raw_ptr, True
            )
        elif np_array.dtype == np.int32:
            raw_ptr = ffi.cast("int*", np_raw_ptr[0])
            ret_val = ffc().flexflow_tensor_get_tensor_int(
                self.handle, ffmodel.handle, raw_ptr, True
            )
        elif np_array.dtype == np.int64:
            raw_ptr = ffi.cast("int64_t*", np_raw_ptr[0])
            ret_val = ffc().flexflow_tensor_get_tensor_int64(
                self.handle, ffmodel.handle, raw_ptr, True
            )
        fflogger.debug(
            "get weights raw_ptr: %s, %s, %s, %s"
            % (str(raw_ptr), str(np_raw_ptr[0]), hex(np_raw_ptr[0]), str(shape))
        )
        assert ret_val == True
        return np_array

    def get_model_output_gradients(self, ffmodel, comm_type):
        shape = self.dims
        if self.data_type == DataType.DT_HALF:
            np_array = np.empty(shape, dtype=np.float16)
        elif self.data_type == DataType.DT_FLOAT:
            np_array = np.empty(shape, dtype=np.float32)
        elif self.data_type == DataType.DT_INT32:
            np_array = np.empty(shape, dtype=np.int32)
        elif self.data_type == DataType.DT_INT64:
            np_array = np.empty(shape, dtype=np.int64)
        else:
            assert 0, f"Unsupported datatype: {self.data_type}"
        np_raw_ptr = np_array.__array_interface__["data"]
        c_comm_type = enum_to_int(ParameterSyncType, comm_type)
        if np_array.dtype == np.float32:
            raw_ptr = ffi.cast("float*", np_raw_ptr[0])
            ret_val = ffc().flexflow_model_get_output_tensor_float(
                ffmodel.handle, self.handle, raw_ptr, True
            )
        else:
            assert 0, "unknown data type"
        fflogger.debug(
            "get weights raw_ptr: %s, %s, %s, %s"
            % (str(raw_ptr), str(np_raw_ptr[0]), hex(np_raw_ptr[0]), str(shape))
        )
        assert ret_val == True
        return np_array

    def get_model_output_tensor(self, ffmodel):
        shape = self.dims
        if self.data_type == DataType.DT_HALF:
            np_array = np.empty(shape, dtype=np.float16)
        elif self.data_type == DataType.DT_FLOAT:
            np_array = np.empty(shape, dtype=np.float32)
        elif self.data_type == DataType.DT_INT32:
            np_array = np.empty(shape, dtype=np.int32)
        elif self.data_type == DataType.DT_INT64:
            np_array = np.empty(shape, dtype=np.int64)
        else:
            assert 0, f"Unsupported datatype: {self.data_type}"
        np_raw_ptr = np_array.__array_interface__["data"]
        if np_array.dtype == np.float32:
            raw_ptr = ffi.cast("float*", np_raw_ptr[0])
            ret_val = ffc().flexflow_model_get_output_tensor_float(
                ffmodel.handle, self.handle, raw_ptr, False
            )
        else:
            assert 0, "unknown data type"
        fflogger.debug(
            "get weights raw_ptr: %s, %s, %s, %s"
            % (str(raw_ptr), str(np_raw_ptr[0]), hex(np_raw_ptr[0]), str(shape))
        )
        assert ret_val == True
        return np_array

    def __get_raw_ptr(self, ffmodel, ffconfig, data_type):
        assert data_type == self.data_type, "Tensor check data type"
        if data_type == DataType.DT_HALF:
            return ffc().flexflow_tensor_get_raw_ptr_float(
                self.handle, ffmodel.handle, ffconfig.handle
            )
        elif data_type == DataType.DT_FLOAT:
            return ffc().flexflow_tensor_get_raw_ptr_float(
                self.handle, ffmodel.handle, ffconfig.handle
            )
        elif data_type == DataType.DT_INT32:
            return ffc().flexflow_tensor_get_raw_ptr_int32(
                self.handle, ffmodel.handle, ffconfig.handle
            )
        else:
            assert 0, "unknown data type"

    def __get_dims(self):
        self.num_dims = ffc().flexflow_tensor_get_num_dims(self.handle)
        # if (self.num_dims == 1):
        #   self.dims = (ffc().flexflow_tensor_get_dim(self.handle, 0),)
        # elif (self.num_dims == 2):
        #   self.dims = (ffc().flexflow_tensor_get_dim(self.handle, 1), ffc().flexflow_tensor_get_dim(self.handle, 0))
        # elif (self.num_dims == 3):
        #   self.dims = (ffc().flexflow_tensor_get_dim(self.handle, 2), ffc().flexflow_tensor_get_dim(self.handle, 1), ffc().flexflow_tensor_get_dim(self.handle, 0))
        # elif (self.num_dims == 4):
        #   self.dims = (ffc().flexflow_tensor_get_dim(self.handle, 3), ffc().flexflow_tensor_get_dim(self.handle, 2), ffc().flexflow_tensor_get_dim(self.handle, 1), ffc().flexflow_tensor_get_dim(self.handle, 0))
        # elif (self.num_dims == 5):
        #   self.dims = (ffc().flexflow_tensor_get_dim(self.handle, 4), ffc().flexflow_tensor_get_dim(self.handle, 3), ffc().flexflow_tensor_get_dim(self.handle, 2), ffc().flexflow_tensor_get_dim(self.handle, 1), ffc().flexflow_tensor_get_dim(self.handle, 0))
        # else:
        #   assert 0, "unknown num_dims"
        d = ffc().flexflow_tensor_get_dims(self.handle)
        if self.num_dims == 1:
            self.dims = (d[0],)
        elif self.num_dims == 2:
            self.dims = (d[1], d[0])
        elif self.num_dims == 3:
            self.dims = (d[2], d[1], d[0])
        elif self.num_dims == 4:
            self.dims = (d[3], d[2], d[1], d[0])
        elif self.num_dims == 5:
            self.dims = (d[4], d[3], d[2], d[1], d[0])
        else:
            assert 0, "unknown num_dims"

    def __get_data_type(self):
        dtype = ffc().flexflow_tensor_get_data_type(self.handle)
        if dtype == 40:
            self.data_type = DataType.DT_BOOLEAN
        elif dtype == 41:
            self.data_type = DataType.DT_INT32
        elif dtype == 42:
            self.data_type = DataType.DT_INT64
        elif dtype == 43:
            self.data_type = DataType.DT_HALF
        elif dtype == 44:
            self.data_type = DataType.DT_FLOAT
        elif dtype == 45:
            self.data_type = DataType.DT_DOUBLE
        else:
            assert 0, "unknown data type {}".format(dtype)

    def __get_owner_op(self, op_type):
        op_handle = ffc().flexflow_tensor_get_owner_op(self.handle)
        if op_handle.impl == ffi.NULL:
            self.owner_op = None
        else:
            self.owner_op = convert_op_handle_to_op(op_type, op_handle)

    def __attach_raw_ptr(self, ffmodel, ffconfig, raw_ptr, column_major=True):
        assert self.mapped == False, "Tensor is already mapped."
        ffc().flexflow_tensor_attach_raw_ptr(
            self.handle, ffmodel.handle, ffconfig.handle, raw_ptr, column_major
        )
        self.mapped = True

    def __detach_raw_ptr(self, ffconfig):
        assert self.mapped == True, "Tensor is not mapped."
        ffc().flexflow_tensor_detach_raw_ptr(self.handle, ffconfig.handle)
        self.mapped = False


# -----------------------------------------------------------------------
# Parameter
# -----------------------------------------------------------------------


class Parameter(Tensor):
    __slots__ = ["parameter_handle"]

    def __init__(self, handle):
        assert ffi.typeof(handle) == ffi.typeof(
            "flexflow_tensor_t"
        ), "Parameter handle is wrong"
        self.parameter_handle = handle
        super(Parameter, self).__init__(self.parameter_handle, deallocate=False)

    def set_weights(self, ffmodel, np_array):
        assert (
            np_array.__array_interface__["strides"] == None
        ), "Parameter set_weights, numpy array strides is not None"
        np_shape = np_array.shape
        num_dims = len(np_shape)
        assert num_dims == self.num_dims, "please check dims (%d == %d)" % (
            num_dims,
            self.num_dims,
        )
        print(np_shape, self.dims)
        for i in range(0, num_dims):
            assert (
                np_shape[i] == self.dims[i]
            ), "please check shape dim %d (%d == %d)" % (i, np_shape[i], self.dims[i])
        c_dims = ffi.new("int[]", self.dims)
        np_raw_ptr = np_array.__array_interface__["data"]
        raw_ptr = ffi.cast("float*", np_raw_ptr[0])
        fflogger.debug(
            "set weights raw_ptr: %s, %s, %s, %s"
            % (str(raw_ptr), str(np_raw_ptr[0]), hex(np_raw_ptr[0]), str(np_shape))
        )
        ret_val = ffc().flexflow_tensor_set_tensor_float(
            self.parameter_handle, ffmodel.handle, num_dims, c_dims, raw_ptr
        )
        assert ret_val == True, ret_val

    def get_weights(self, ffmodel):
        shape = self.dims
        np_array = np.empty(shape, dtype=np.float32)
        np_raw_ptr = np_array.__array_interface__["data"]
        raw_ptr = ffi.cast("float*", np_raw_ptr[0])
        fflogger.debug(
            "get weights raw_ptr: %s, %s, %s, %s"
            % (str(raw_ptr), str(np_raw_ptr[0]), hex(np_raw_ptr[0]), str(shape))
        )
        ret_val = ffc().flexflow_tensor_get_tensor_float(
            self.parameter_handle, ffmodel.handle, raw_ptr, False
        )
        assert ret_val == True
        return np_array


# -----------------------------------------------------------------------
# FFModel
# -----------------------------------------------------------------------


class FFModel(object):
    """ """

    __slots__ = [
        "handle",
        "_handle",
        "_layers",
        "_nb_layers",
        "_ffconfig",
        "_tracing_id",
        "initializers",
        "attr_tensors",
    ]

    def __init__(self, ffconfig):
        """Constructor of FFModel.

        :param ffconfig: configurations of FlexFlow and the created model.
        :type ffconfig: FFConfig

        :returns:  FFModel -- the model.
        """
        self.handle = ffc().flexflow_model_create(ffconfig.handle, ffconfig.cpu_offload)
        self._handle = ffi.gc(self.handle, ffc().flexflow_model_destroy)
        self._layers = dict()
        self._nb_layers = 0
        self._ffconfig = ffconfig
        global ff_tracing_id
        self._tracing_id = ff_tracing_id
        ff_tracing_id += 1
        self.initializers = {}
        self.attr_tensors = {}

    def get_layers(self):
        return self._layers

    def add_layer(self, op_type, name):
        layer_id = self._nb_layers
        op_handle = ffc().flexflow_model_get_last_layer(self.handle)
        self._layers[self._nb_layers] = convert_op_handle_to_op(
            op_type, op_handle, idx=layer_id, name=name
        )
        self._nb_layers += 1

    def create_tensor(self, dims, data_type, create_grad=True):
        """Instantiate a FlexFlow tensor.

        :param x: a shape tuple/list (integers), including the batch size.
        :type x: list of int

        :param data_type: the datatype of the created tensor. Options are
          DT_FLOAT, DT_DOUBLE, DT_INT32, DT_INT64, DT_BOOLEAN.
        :type data_type: DataType

        :param create_grad: weather the tensor creates a gradients vector.
          If you don't specify anything, a gradients vector is used.
        :type create_grad: bool

        :returns:  Tensor -- the output tensor.
        """
        c_dims = ffi.new("int[]", dims)
        c_data_type = enum_to_int(DataType, data_type)
        num_dims = len(dims)
        handle = ffc().flexflow_tensor_create(
            self.handle, num_dims, c_dims, c_data_type, create_grad
        )
        return Tensor(handle)

    def map_tensor(self, tensor, parallel_op=None):
        op_handle = self.__get_op_handle(parallel_op)
        ffc().flexflow_tensor_map(self.handle, tensor.handle, op_handle)

    def create_constant(self, dims, value, data_type):
        c_dims = ffi.new("int[]", dims)
        c_data_type = enum_to_int(DataType, data_type)
        num_dims = len(dims)
        handle = ffc().flexflow_constant_create(
            self.handle, num_dims, c_dims, value, c_data_type
        )
        return Tensor(handle)

    def exp(self, x, name=None):
        """Exponential activation function.

        :param x: the input Tensor.
        :type x: Tensor

        :param name: the name of the layer. Default is None.
        :type name: string

        :returns:  Tensor -- the output tensor.
        """
        c_name = get_c_name(name)
        handle = ffc().flexflow_model_add_exp(self.handle, x.handle, c_name)
        self.add_layer(OpType.EXP, name)
        return Tensor(handle, owner_op_type=OpType.EXP)

    def sin(self, x, name=None):
        """Elementwise sine function.

        :param x: the input Tensor.
        :type x: Tensor

        :param name: the name of the layer. Default is None.
        :type name: string

        :returns:  Tensor -- the output tensor.
        """
        c_name = get_c_name(name)
        handle = ffc().flexflow_model_add_sin(self.handle, x.handle, c_name)
        self.add_layer(OpType.SIN, name)
        return Tensor(handle, owner_op_type=OpType.SIN)

    def cos(self, x, name=None):
        """Elementwise cosine function.

        :param x: the input Tensor.
        :type x: Tensor

        :param name: the name of the layer. Default is None.
        :type name: string

        :returns:  Tensor -- the output tensor.
        """
        c_name = get_c_name(name)
        handle = ffc().flexflow_model_add_cos(self.handle, x.handle, c_name)
        self.add_layer(OpType.COS, name)
        return Tensor(handle, owner_op_type=OpType.COS)

    def add(self, x, y, inplace_a=False, name=None):
        """Layer that adds two input Tensors, :attr:`output = x + y`.

        :param x: the first input Tensor.
        :type x: Tensor

        :param y: the second input Tensor.
        :type y: Tensor

        :param name: the name of the layer. Default is None.
        :type name: string

        :returns:  Tensor -- the output tensor.
        """
        c_name = get_c_name(name)
        handle = ffc().flexflow_model_add_add(
            self.handle, x.handle, y.handle, inplace_a, c_name
        )
        self.add_layer(OpType.ADD, name)
        return Tensor(handle, owner_op_type=OpType.ADD)

    def subtract(self, x, y, inplace_a=False, name=None):
        """Layer that subtracts two input Tensors, :attr:`output = x * y`.

        :param x: the first input Tensor.
        :type x: Tensor

        :param y: the second input Tensor.
        :type y: Tensor

        :param name: the name of the layer. Default is None.
        :type name: string

        :returns:  Tensor -- the output tensor.
        """
        c_name = get_c_name(name)
        handle = ffc().flexflow_model_add_subtract(
            self.handle, x.handle, y.handle, inplace_a, c_name
        )
        self.add_layer(OpType.SUBTRACT, name)
        return Tensor(handle, owner_op_type=OpType.SUBTRACT)

    def multiply(self, x, y, inplace_a=False, name=None):
        """Layer that multiplies (element-wise) two input Tensors, :attr:`output = x * y`.

        :param x: the first input Tensor.
        :type x: Tensor

        :param y: the second input Tensor.
        :type y: Tensor

        :param name: the name of the layer. Default is None.
        :type name: string

        :returns:  Tensor -- the output tensor.
        """
        c_name = get_c_name(name)
        handle = ffc().flexflow_model_add_multiply(
            self.handle, x.handle, y.handle, inplace_a, c_name
        )
        self.add_layer(OpType.MULTIPLY, name)
        return Tensor(handle, owner_op_type=OpType.MULTIPLY)

    def divide(self, x, y, inplace_a=False, name=None):
        """Layer that divides (element-wise) two input Tensors, :attr:`output = x / y`.

        :param x: the first input Tensor.
        :type x: Tensor

        :param y: the second input Tensor.
        :type y: Tensor

        :param name: the name of the layer. Default is None.
        :type name: string

        :returns:  Tensor -- the output tensor.
        """
        c_name = get_c_name(name)
        handle = ffc().flexflow_model_add_divide(
            self.handle, x.handle, y.handle, inplace_a, c_name
        )
        self.add_layer(OpType.DIVIDE, name)
        return Tensor(handle, owner_op_type=OpType.DIVIDE)

    def max(self, x, y, inplace_a=False, name=None):
        """Layer that computes the max (element-wise) two input Tensors, :attr:`output = max(x,y)`.

        :param x: the first input Tensor.
        :type x: Tensor

        :param y: the second input Tensor.
        :type y: Tensor

        :param name: the name of the layer. Default is None.
        :type name: string

        :returns:  Tensor -- the output tensor.
        """
        c_name = get_c_name(name)
        handle = ffc().flexflow_model_add_max(
            self.handle, x.handle, y.handle, inplace_a, c_name
        )
        self.add_layer(OpType.MAX, name)
        return Tensor(handle, owner_op_type=OpType.MAX)

    def min(self, x, y, inplace_a=False, name=None):
        """Layer that computes the min (element-wise) two input Tensors, :attr:`output = min(x,y)`.

        :param x: the first input Tensor.
        :type x: Tensor

        :param y: the second input Tensor.
        :type y: Tensor

        :param name: the name of the layer. Default is None.
        :type name: string

        :returns:  Tensor -- the output tensor.
        """
        c_name = get_c_name(name)
        handle = ffc().flexflow_model_add_min(
            self.handle, x.handle, y.handle, inplace_a, c_name
        )
        self.add_layer(OpType.MIN, name)
        return Tensor(handle, owner_op_type=OpType.MIN)

    def reduce_sum(self, input, axes, keepdims=False, name=None):
        """Layer that computes the sum of the input Tensor along given axes.

        :param input: the input Tensor.
        :type input: Tensor

        :param axes: the axes along which reduction is applied
        :type axes: List[int]

        :param name: the name of the layer. Default is None.
        :type name: string

        :returns:  Tensor -- the output tensor.
        """
        c_name = get_c_name(name)
        c_axes = ffi.new("int[]", axes)
        handle = ffc().flexflow_model_add_reduce_sum(
            self.handle, input.handle, c_axes, len(axes), keepdims, c_name
        )
        self.add_layer(OpType.REDUCE_SUM, name)
        return Tensor(handle, owner_op_type=OpType.REDUCE_SUM)

    def rsqrt(self, input, name=None):
        """Layer that computes the element-wise reciprocal square-root.

        :param input: the input Tensor.
        :type input: Tensor

        :param name: the name of the layer. Default is None.
        :type name: string

        :returns:  Tensor -- the output tensor.
        """
        c_name = get_c_name(name)
        handle = ffc().flexflow_model_add_rsqrt(self.handle, input.handle, c_name)
        self.add_layer(OpType.RSQRT, name)
        return Tensor(handle, owner_op_type=OpType.RSQRT)

    def pow(self, input, exponent, name=None):
        """Layer that computes the element-wise power.

        :param input: the input Tensor.
        :type input: Tensor

        :param exponent: exponent to raise each element in the input tensor.
        :type exponent: float

        :param name: the name of the layer. Default is None.
        :type name: string

        :returns:  Tensor -- the output tensor.
        """
        c_name = get_c_name(name)
        handle = ffc().flexflow_model_add_pow(
            self.handle, input.handle, exponent, c_name
        )
        self.add_layer(OpType.POW, name)
        return Tensor(handle, owner_op_type=OpType.POW)

    def mean(self, input, dims, keepdims=False, name=None):
        """Layer that computes the mean of the input tensor across the given
        dimensions.

        :param input: the input Tensor.
        :type input: Tensor

        :param dims: dimensions to take the mean over.
        :type dims: list

        :param keepdims: keeps the dimensions in :attr:`dims` as size 1 if True and
                         collapses the dimension if False. Default is False.
        :type keepdims: bool

        :param name: the name of the layer. Default is None.
        :type name: string

        :returns:  Tensor -- the output tensor.
        """
        dims = list(dims)
        c_dims = ffi.new("int[]", dims)
        c_name = get_c_name(name)
        handle = ffc().flexflow_model_add_mean(
            self.handle, input.handle, c_dims, len(dims), keepdims, c_name
        )
        self.add_layer(OpType.MEAN, name)
        return Tensor(handle, owner_op_type=OpType.MEAN)

    def conv2d(
        self,
        input,
        out_channels,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        activation=ActiMode.AC_MODE_NONE,
        groups=1,
        use_bias=True,
        shared_op=None,
        kernel_initializer=None,
        bias_initializer=None,
        name=None,
    ):
        """This layer creates a 2D convolution kernel that is convolved with the layer :attr:`input`
        to produce a tensor of :attr:`output`.

        The size of input tensor is :math:`(N, C_{in}, H, W)` and the size of output tensor
        is :math:`(N, C_{out}, H_{out}, W_{out})`, which can be calculated by:

        .. math::
          C_{out} = out\_channels

        .. math::
          K_{H} = kernel\_h

        .. math::
          K_{W} = kernel\_w

        .. math::
          S_{H} = stride\_h

        .. math::
          S_{W} = stride\_w

        .. math::
          P_{H} = padding\_h

        .. math::
          P_{S} = padding\_s

        .. math::
          H_{out} = (H - K_{H} + 2 * P_{H}) / S_{H} + 1

        .. math::
          W_{out} = (W - K_{W} + 2 * P_{W}) / S_{W} + 1

        :param input: the input Tensor.
        :type input: Tensor

        :param out\_channels: the dimensionality of the output space (i.e. the number of output filters in the convolution).
        :type out\_channels: int

        :param kernel_h: the height of the 2D convolution window: :math:`K_{H}`.
        :type kernel_h: int

        :param kernel_w: the width of the 2D convolution window: :math:`K_{W}`.
        :type kernel_w: int

        :param stride_h: the stride of the convolution along the height: :math:`S_{H}`.
        :type stride_h: int

        :param stride_w: the stride of the convolution along the width: :math:`S_{W}`.
        :type stride_w: int

        :param padding_h: the amount of implicit zero-paddings along the height: :math:`P_{H}`.
        :type padding_h: int

        :param padding_w: the amount of implicit zero-paddings along the width: :math:`P_{W}`.
        :type padding_w: int

        :param activation: Activation function to use. Default is ActiMode.AC_MODE_NONE.
        :type activation: ActiMode

        :param groups: the number of groups in this convolution
        :type groups: int

        :param use_bias: whether the layer uses a bias vector. Default is True.
        :type use_bias: bool

        :param shared_op: the layer whose parameters are shared with. Default is None.
        :type shared_op: Op

        :param kernel_initializer: Initializer for the kernel weights matrix. If it is set to None, the GlorotUniformInitializer is applied.
        :type kernel_initializer: Initializer

        :param bias_initializer: Initializer for the bias vector. If it is set to None, the ZeroInitializer is applied.
        :type bias_initializer: Initializer

        :param name: the name of the layer. Default is None.
        :type name: string

        :returns:  Tensor -- the output tensor.
        """
        shared_op_handle = self.__get_op_handle(shared_op)
        c_activation = enum_to_int(ActiMode, activation)
        kernel_init_handle = self.__get_initializer_handle(kernel_initializer)
        bias_init_handle = self.__get_initializer_handle(bias_initializer)
        c_name = get_c_name(name)
        handle = ffc().flexflow_model_add_conv2d(
            self.handle,
            input.handle,
            out_channels,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            padding_h,
            padding_w,
            c_activation,
            groups,
            use_bias,
            shared_op_handle,
            kernel_init_handle,
            bias_init_handle,
            c_name,
        )
        self.add_layer(OpType.CONV2D, name)
        return Tensor(handle, owner_op_type=OpType.CONV2D)

    def embedding(
        self,
        input,
        num_embeddings,
        embedding_dim,
        aggr,
        dtype=DataType.DT_FLOAT,
        shared_op=None,
        kernel_initializer=None,
        name=None,
    ):
        """Layer that turns positive integers into dense vectors of fixed size

        :param input: the input Tensor.
        :type input: Tensor

        :param num_embeddings: size of the vocabulary, i.e. maximum integer index + 1
        :type num_embeddings: int

        :param embedding_dim: dimension of the dense embedding.
        :type embedding_dim: int

        :param aggr: aggregation mode. Options are AGGR_MODE_NONE, AGGR_MODE_SUM and AGGR_MODE_AVG.
        :type aggr: AggrMode

        :param dtype: the tensor data type. Options are DT_BOOLEAN, DT_INT32, DT_INT64, DT_HALF, DT_FLOAT, DT_DOUBLE, DT_INT4, DT_INT8, DT_NONE
        :type dtype: DataType

        :param shared_op: the layer whose parameters are shared with. Default is None.
        :type shared_op: Op

        :param kernel_initializer: Initializer for the kernel weights matrix. If it is set to None, the GlorotUniformInitializer is applied.
        :type kernel_initializer: Initializer

        :param name: the name of the layer. Default is None.
        :type name: string

        :returns:  Tensor -- the output tensor.
        """
        c_name = get_c_name(name)
        shared_op_handle = self.__get_op_handle(shared_op)
        c_aggr = enum_to_int(AggrMode, aggr)
        c_dtype = enum_to_int(DataType, dtype)
        if kernel_initializer is None:
            kernel_initializer = GlorotUniformInitializer(42)
        assert (
            (type(kernel_initializer) is GlorotUniformInitializer)
            or (type(kernel_initializer) is ZeroInitializer)
            or (type(kernel_initializer) is UniformInitializer)
            or (type(kernel_initializer) is NormInitializer)
        ), f"Unknown initializer type: {kernel_initializer}"
        handle = ffc().flexflow_model_add_embedding(
            self.handle,
            input.handle,
            num_embeddings,
            embedding_dim,
            c_aggr,
            c_dtype,
            shared_op_handle,
            kernel_initializer.handle,
            c_name,
        )
        # NOTE: We must keep a reference to the initializer or else it will be
        # immediately destructed
        self.initializers[name] = kernel_initializer
        self.add_layer(OpType.EMBEDDING, name)
        return Tensor(handle, owner_op_type=OpType.EMBEDDING)

    def pool2d(
        self,
        input,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        pool_type=PoolType.POOL_MAX,
        activation=ActiMode.AC_MODE_NONE,
        name=None,
    ):
        """Pooling operation for 2D spatial data.

        The size of input tensor is :math:`(N, C_{in}, H, W)` and the size of output tensor
        is :math:`(N, C_{out}, H_{out}, W_{out})`, which can be calculated by:

        .. math::
          C_{out} = out\_channels

        .. math::
          K_{H} = kernel\_h

        .. math::
          K_{W} = kernel\_w

        .. math::
          S_{H} = stride\_h

        .. math::
          S_{W} = stride\_w

        .. math::
          P_{H} = padding\_h

        .. math::
          P_{S} = padding\_s

        .. math::
          H_{out} = (H - K_{H} + 2 * P_{H}) / S_{H} + 1

        .. math::
          W_{out} = (W - K_{W} + 2 * P_{W}) / S_{W} + 1

        :param input: the input Tensor.
        :type input: Tensor

        :param kernel_h: the height of the 2D pooling window: :math:`K_{H}`.
        :type kernel_h: int

        :param kernel_w: the width of the 2D pooling window: :math:`K_{W}`.
        :type kernel_w: int

        :param stride_h: the stride of the pooling along the height: :math:`S_{H}`.
        :type stride_h: int

        :param stride_w: the stride of the pooling along the width: :math:`S_{W}`.
        :type stride_w: int

        :param padding_h: the amount of implicit zero-paddings along the height: :math:`P_{H}`.
        :type padding_h: int

        :param padding_w: the amount of implicit zero-paddings along the width: :math:`P_{W}`.
        :type padding_w: int

        :param activation: Tyoe of pooling function to use. If you don't specify anything, PoolType.POOL_MAX is applied.
        :type activation: PoolType

        :param activation: Activation function to use. Default is ActiMode.AC_MODE_NONE.
        :type activation: ActiMode

        :param name: the name of the layer. Default is None.
        :type name: string

        :returns:  Tensor -- the output tensor.
        """
        c_name = get_c_name(name)
        c_pool_type = enum_to_int(PoolType, pool_type)
        c_activation = enum_to_int(ActiMode, activation)
        handle = ffc().flexflow_model_add_pool2d(
            self.handle,
            input.handle,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            padding_h,
            padding_w,
            c_pool_type,
            c_activation,
            c_name,
        )
        self.add_layer(OpType.POOL2D, name)
        return Tensor(handle, owner_op_type=OpType.POOL2D)

    def batch_norm(self, input, relu=True, name=None):
        """Layer that normalizes its inputs.

        Batch normalization applies a transformation that maintains the mean output close to 0 and the output standard deviation close to 1.

        :param input: the list of input Tensors.
        :type input: Tensor

        :param relu: whether a ReLU function is applied. Default is True.
        :type relu: bool

        :param name: the name of the layer. Default is None.
        :type name: string

        :returns:  Tensor -- the output tensor.
        """
        c_name = get_c_name(name)
        handle = ffc().flexflow_model_add_batch_norm(
            self.handle, input.handle, relu, c_name
        )
        self.add_layer(OpType.BATCH_NORM, name)
        return Tensor(handle, owner_op_type=OpType.BATCH_NORM)

    def layer_norm(
        self, input, axes, elementwise_affine=True, eps=1e-5, use_bias=True, name=None
    ):
        """Add a LayerNorm layer

        :param input: The input tensor
        :type input: Tensor
        :param axes: Indicate which axes (starting from the end) the LayerNorm should normalize over
        :type axes: Union[int, List[int]]
        :param elementwise_affine: Whether the LayerNorm should use the gamma weight for scaling, defaults to True
        :type elementwise_affine: bool, optional
        :param eps: A small float value added to the LayerNorm denominator for numerical stability, defaults to 1e-5
        :type eps: float, optional
        :param use_bias: Whether to add a beta bias to the LayerNorm result, defaults to True
        :type use_bias: bool, optional
        :param name: Name of the operator, also used for loading weights in inference mode, defaults to None
        :type name: _type_, optional
        :return: The LayerNorm output tensor
        :rtype: Tensor
        """
        c_name = get_c_name(name)
        c_axes = ffi.new("int[]", axes)
        handle = ffc().flexflow_model_add_layer_norm(
            self.handle,
            input.handle,
            len(axes),
            c_axes,
            elementwise_affine,
            eps,
            use_bias,
            c_name,
        )
        self.add_layer(OpType.LAYER_NORM, name)
        return Tensor(handle, owner_op_type=OpType.LAYER_NORM)

    def residual_layer_norm(
        self,
        input,
        residual1,
        residual2,
        use_two_residuals,
        axes,
        elementwise_affine=True,
        eps=1e-5,
        use_bias=True,
        inplace_residual=False,
        name=None,
    ):
        """Add a fused LayerNorm + Residual layer. This operator uses a single kernel, resulting in 
        better efficiency compared to using separate element-wise add and LayerNorm operators.

        :param input: The input tensor
        :type input: Tensor
        :param residual1: The residual tensor to add to the input before computing the LayerNorm
        :type residual1: Tensor
        :param residual2: An optional second residual tensor to add to the input (in addition to residual1) before computing the LayerNorm
        :type residual2: Tensor
        :param use_two_residuals: A boolean that should be set to True if using the second optional residual, False otherwise
        :type use_two_residuals: bool
        :param axes: Indicate which axes (starting from the end) the LayerNorm should normalize over
        :type axes: List[int]
        :param elementwise_affine: Whether the LayerNorm should use the gamma weight for scaling, defaults to True
        :type elementwise_affine: bool, optional
        :param eps: A small float value added to the LayerNorm denominator for numerical stability, defaults to 1e-5
        :type eps: float, optional
        :param use_bias: Whether to add a beta bias to the LayerNorm result, defaults to True
        :type use_bias: bool, optional
        :param inplace_residual: Whether to perform the residual computation inplace in the input tensor, defaults to False
        :type inplace_residual: bool, optional
        :param name: Name of the operator, also used for loading weights in inference mode, defaults to None
        :type name: str, optional
        :return: A tensor with the sum of the input and residual(s), and the LayerNorm output
        :rtype: (Tensor, Tensor)
        """
        c_name = get_c_name(name)
        c_axes = ffi.new("int[]", axes)
        residual2_handle = (
            residual1.handle
        )  # This is intentional. Data will be ignored, and we cannot pass None
        if use_two_residuals:
            assert residual2 is not None
            residual2_handle = residual2.handle
        handles_array = ffc().flexflow_model_add_residual_layer_norm(
            self.handle,
            input.handle,
            residual1.handle,
            residual2_handle,
            use_two_residuals,
            len(axes),
            c_axes,
            elementwise_affine,
            eps,
            use_bias,
            inplace_residual,
            c_name,
        )
        self.add_layer(OpType.RESIDUAL_LAYERNORM, name)
        return Tensor(
            handles_array[0], owner_op_type=OpType.RESIDUAL_LAYERNORM
        ), Tensor(handles_array[1], owner_op_type=OpType.RESIDUAL_LAYERNORM)

    def add_bias_residual_layer_norm(
        self,
        input,
        residual,
        axes,
        elementwise_affine=True,
        eps=1e-5,
        use_bias=True,
        inplace_residual=False,
        name=None,
    ):
        """Add a Attention Bias + Residual + LayerNorm layer. This operator uses a single kernel, 
        resulting in better efficiency compared to using separate attention bias addition + 
        element-wise residual addition + LayerNorm operators.

        :param input: The input tensor
        :type input: Tensor
        :param residual: The residual tensor
        :type residual: Tensor
        :param axes: Indicate which axes (starting from the end) the LayerNorm should normalize over
        :type axes: Union[int, List[int]]
        :param elementwise_affine: Whether the LayerNorm should use the gamma weight for scaling, defaults to True
        :type elementwise_affine: bool, optional
        :param eps: A small float value added to the LayerNorm denominator for numerical stability, defaults to 1e-5
        :type eps: float, optional
        :param use_bias: Whether to add a beta bias to the LayerNorm result, defaults to True
        :type use_bias: bool, optional
        :param inplace_residual: Whether to perform the residual computation inplace in the input tensor, defaults to False
        :type inplace_residual: bool, optional
        :param name: Name of the operator, also used for loading weights in inference mode, defaults to None
        :type name: _type_, optional
        :return: A tensor with the sum of the attention bias, input and residual(s), and the LayerNorm output
        :rtype: (Tensor, Tensor)
        """
        c_name = get_c_name(name)
        c_axes = ffi.new("int[]", axes)
        handles_array = ffc().flexflow_model_add_add_bias_residual_layer_norm(
            self.handle,
            input.handle,
            residual.handle,
            len(axes),
            c_axes,
            elementwise_affine,
            eps,
            use_bias,
            inplace_residual,
            c_name,
        )
        self.add_layer(OpType.ADD_BIAS_RESIDUAL_LAYERNORM, name)
        return Tensor(
            handles_array[0], owner_op_type=OpType.ADD_BIAS_RESIDUAL_LAYERNORM
        ), Tensor(handles_array[1], owner_op_type=OpType.ADD_BIAS_RESIDUAL_LAYERNORM)

    def sigmoid_silu_multi(self, input1, input2, name=None):
        c_name = get_c_name(name)
        handle = ffc().flexflow_model_add_sigmoid_silu_multi(
            self.handle, input1.handle, input2.handle, c_name
        )
        self.add_layer(OpType.SIGMOID_SILU_MULTI, name)
        return Tensor(handle, owner_op_type=OpType.SIGMOID_SILU_MULTI)

    def batch_matmul(
        self, A, B, a_seq_length_dim=None, b_seq_length_dim=None, name=None
    ):
        """Layer that applied batched matrix multiplication onto two input Tensors, :attr:`output = x * y`.

        :param A: the first input Tensor.
        :type A: Tensor

        :param B: the second input Tensor.
        :type B: Tensor

        :param a_seq_length_dim: an int when set indicating the a_seq_length_dim dimention of A is a sequence_length dimension
        :type a_seq_length_dim: int

        :param b_seq_length_dim: an int when set indicating the b_seq_length_dim dimention of B is a sequence_length dimension
        :type b_seq_length_dim: int

        :param name: the name of the layer. Default is None.
        :type name: string

        :param name:  Whether to add use bias in layer normalization
        :type name: bool

        :returns:  Tensor -- the output tensor.
        """
        if a_seq_length_dim is None:
            a_seq_length_dim = -1
        if b_seq_length_dim is None:
            b_seq_length_dim = -1
        handle = ffc().flexflow_model_add_batch_matmul(
            self.handle, A.handle, B.handle, a_seq_length_dim, b_seq_length_dim
        )
        self.add_layer(OpType.BATCH_MATMUL, name)
        return Tensor(handle, owner_op_type=OpType.BATCH_MATMUL)

    def dense(
        self,
        input,
        out_dim,
        activation=ActiMode.AC_MODE_NONE,
        use_bias=True,
        datatype=DataType.DT_NONE,
        shared_op=None,
        kernel_initializer=None,
        bias_initializer=None,
        kernel_regularizer=None,
        name=None,
    ):
        """Dense implements the operation: :attr:`output = activation(dot(input, kernel) + bias)` where
        :attr:`activation` is the element-wise activation function passed as the activation argument,
        :attr:`kernel` is a weights matrix created by the layer, and
        :attr:`bias` is a bias vector created by the layer (only applicable if :attr:`use_bias` is True).

        The size of input tensor is :math:`(N, C_{in})` and the size of output tensor
        is :math:`(N, C_{out})`, where :math:`C_{out} = out\_dim`

        :param input: the input Tensor.
        :type input: Tensor

        :param out\_dim: dimensionality of the output space.
        :type out\_dim: int

        :param activation: Activation function to use. Default is ActiMode.AC_MODE_NONE.
        :type activation: ActiMode

        :param use_bias: whether the layer uses a bias vector. Default is True.
        :type use_bias: bool

        :param shared_op: the layer whose parameters are shared with. Default is None.
        :type shared_op: Op

        :param kernel_initializer: Initializer for the kernel weights matrix. If it is set to None, the GlorotUniformInitializer is applied.
        :type kernel_initializer: Initializer

        :param bias_initializer: Initializer for the bias vector. If it is set to None, the ZeroInitializer is applied.
        :type bias_initializer: Initializer

        :param kernel_regularizer: Regularizer for the kernel weights matrix
        :type bias_initializer: Regularizer

        :param name: the name of the layer. Default is None.
        :type name: string

        :returns:  Tensor -- the output tensor.
        """
        c_name = get_c_name(name)
        shared_op_handle = self.__get_op_handle(shared_op)
        c_activation = enum_to_int(ActiMode, activation)
        c_datatype = enum_to_int(DataType, datatype)
        kernel_init_handle = self.__get_initializer_handle(kernel_initializer)
        bias_init_handle = self.__get_initializer_handle(bias_initializer)
        if kernel_regularizer:
            c_kernel_reg_type = enum_to_int(RegularizerMode, kernel_regularizer.type)
            kernel_reg_lambda = kernel_regularizer._lambda
        else:
            c_kernel_reg_type = enum_to_int(
                RegularizerMode, RegularizerMode.REG_MODE_NONE
            )
            kernel_reg_lambda = 0.0
        handle = ffc().flexflow_model_add_dense(
            self.handle,
            input.handle,
            out_dim,
            c_activation,
            use_bias,
            c_datatype,
            shared_op_handle,
            kernel_init_handle,
            bias_init_handle,
            c_kernel_reg_type,
            kernel_reg_lambda,
            c_name,
        )
        self.add_layer(OpType.LINEAR, name)
        return Tensor(handle, owner_op_type=OpType.LINEAR)

    def concat(self, tensors, axis, name=None):
        """Layer that concatenates a list of inputs.

        It takes as input a list of tensors, all of the same shape except for the concatenation axis, and returns a single tensor that is the concatenation of all inputs.

        :param input: the list of input Tensors.
        :type input: List of Tensors

        :param axis: the dimension along which to concatenate.
        :type axis: int

        :param name: the name of the layer. Default is None.
        :type name: string

        :returns:  Tensor -- the output tensor.
        """
        assert type(tensors) is list, "tensors should be a list"
        tensor_handle_list = []
        n = len(tensors)
        assert n <= 256, "Please increase MAX_NUM_INPUTS"
        for tensor in tensors:
            tensor_handle_list.append(tensor.handle)
        c_tensor_handle_list = ffi.new("flexflow_tensor_t[]", tensor_handle_list)
        c_name = get_c_name(name)
        handle = ffc().flexflow_model_add_concat(
            self.handle, n, c_tensor_handle_list, axis, c_name
        )
        self.add_layer(OpType.CONCAT, name)
        return Tensor(handle, owner_op_type=OpType.CONCAT)

    def split(self, input, sizes, axis, name=None):
        """Layer that splits a :attr:`input` tensor into a list of tensors.

        :param input: the input Tensor.
        :type input: Tensor

        :param sizes: either an int indicating the number of splits along axis or a Python list containing the sizes of each output tensor along axis. If a scalar, then it must evenly divide :attr:`input.dims[axis]`; otherwise the sum of sizes along the split axis must match that of the :attr:`input`.
        :type sizes: int or list of int

        :param axis: the dimension along which to split.
        :type axis: int

        :param name: the name of the layer. Default is None.
        :type name: string

        :returns:  list of Tensors -- the output tensors.
        """
        if type(sizes) is list:
            split = sizes
        else:
            assert input.dims[axis] % sizes == 0, "Split dimension is not divisible"
            split = [input.dims[axis] // sizes for i in range(sizes)]
        n = len(split)
        assert n <= 256, "Please increase MAX_NUM_OUTPUTS"
        c_split = ffi.new("int[]", split)
        c_outputs_handle_list = ffi.new("flexflow_tensor_t[256]")
        c_name = get_c_name(name)
        ffc().flexflow_model_add_split(
            self.handle, input.handle, n, c_outputs_handle_list, c_split, axis, c_name
        )
        output_tensor_list = []
        for i in range(n):
            tensor_p_handle = ffi.new("flexflow_tensor_t*")
            tensor_p_handle.impl = c_outputs_handle_list[i].impl
            output_tensor_list.append(
                Tensor(None, owner_op_type=OpType.SPLIT, p_handle=tensor_p_handle)
            )
        self.add_layer(OpType.SPLIT, name)
        del c_outputs_handle_list
        return output_tensor_list

    def flat(self, input, name=None):
        """Flattens the input. Does not affect the batch size.

        :param input: the input Tensor.
        :type input: Tensor

        :param name: the name of the layer. Default is None.
        :type name: string

        :returns:  Tensor -- the output tensor.
        """
        c_name = get_c_name(name)
        handle = ffc().flexflow_model_add_flat(self.handle, input.handle, c_name)
        self.add_layer(OpType.FLAT, name)
        return Tensor(handle, owner_op_type=OpType.FLAT)

    def softmax(self, input, axis=-1, name=None):
        """Softmax activation function.

        :param input: the input Tensor.
        :type input: Tensor

        :param name: the name of the layer. Default is None.
        :type name: string

        :returns:  Tensor -- the output tensor.
        """
        c_name = get_c_name(name)
        handle = ffc().flexflow_model_add_softmax(
            self.handle, input.handle, axis, c_name
        )
        self.add_layer(OpType.SOFTMAX, name)
        return Tensor(handle, owner_op_type=OpType.SOFTMAX)

    def reshape(self, input, shape, name=None):
        """Layer that reshapes inputs into the given shape.

        Given a :attr:`input` tensor, this operation returns a output tensor that has the same values as tensor in the same order,
        except with a new shape given by :attr:`shape`.

        :param input: the input Tensor.
        :type input: Tensor

        :param shape: A list defining the shape of the output tensor.
        :type shape: list of int

        :param name: the name of the layer. Default is None.
        :type name: string

        :returns:  Tensor -- the output tensor.
        """
        c_name = get_c_name(name)
        c_shape = ffi.new("int[]", shape)
        handle = ffc().flexflow_model_add_reshape(
            self.handle, input.handle, len(shape), c_shape, c_name
        )
        self.add_layer(OpType.RESHAPE, name)
        return Tensor(handle, owner_op_type=OpType.RESHAPE)

    def gather(self, input, index, dim, name=None):
        """Layer that gathers values along the dim axis.

        :param input: the input tensor
        :type input: Tensor

        :param index: the index tensor, which specifies the indices of elements to gather
        :type index: Tensor

        :param dim: the axis along which to index
        :type dim: int

        :param name: the name of the layer. Default is None
        :type name: string

        :returns: Tensor -- the output tensor
        """
        c_name = get_c_name(name)
        handle = ffc().flexflow_model_add_gather(
            self.handle, input.handle, index.handle, dim, c_name
        )
        self.add_layer(OpType.GATHER, name)
        return Tensor(handle, owner_op_type=OpType.GATHER)

    def transpose(self, input, perm, name=None):
        """Transposes the :attr:`input` tensor. Permutes the dimensions according to perm

        :param input: the input Tensor.
        :type input: Tensor

        :param perm: A permutation of the dimensions of a.
        :type perm: List of int

        :param name: the name of the layer. Default is None.
        :type name: string

        :returns:  Tensor -- the output tensor.
        """
        c_name = get_c_name(name)
        c_perm = ffi.new("int[]", perm)
        handle = ffc().flexflow_model_add_transpose(
            self.handle, input.handle, len(perm), c_perm, c_name
        )
        self.add_layer(OpType.TRANSPOSE, name)
        return Tensor(handle, owner_op_type=OpType.TRANSPOSE)

    def reverse(self, input, axis, name=None):
        """Layer that reverses specific dimensions of a tensor.

        Given a :attr:`input` tensor, this operation reverses the dimension :attr:`axis`.

        :param input: the input Tensor.
        :type input: Tensor

        :param axis: the dimension to reverse.
        :type axis: int

        :param name: the name of the layer. Default is None.
        :type name: string

        :returns:  Tensor -- the output tensor.
        """
        c_name = get_c_name(name)
        handle = ffc().flexflow_model_add_reverse(
            self.handle, input.handle, axis, c_name
        )
        self.add_layer(OpType.REVERSE, name)
        return Tensor(handle, owner_op_type=OpType.REVERSE)

    def scalar_multiply(self, input, scalar, inplace=True, name=None):
        """Scalar multiplication of a tensor by an scalar.

        :param input: the input Tensor.
        :type input: Tensor

        :param input: the scalar
        :type scalar: float

        :param name: the name of the layer. Default is None.
        :type name: string

        :returns:  Tensor -- the output tensor.
        """
        c_name = get_c_name(name)
        handle = ffc().flexflow_model_add_scalar_multiply(
            self.handle, input.handle, scalar, inplace, c_name
        )
        self.add_layer(OpType.SCALAR_MULTIPLY, name)
        return Tensor(handle, owner_op_type=OpType.SCALAR_MULTIPLY)

    def scalar_add(self, input, scalar, inplace=True, name=None):
        """Scalar addition of a scalar to each entry of a tensor.

        :param input: the input Tensor.
        :type input: Tensor

        :param input: the scalar
        :type scalar: float

        :param name: the name of the layer. Default is None.
        :type name: string

        :returns:  Tensor -- the output tensor.
        """
        c_name = get_c_name(name)
        handle = ffc().flexflow_model_add_scalar_add(
            self.handle, input.handle, scalar, inplace, c_name
        )
        self.add_layer(OpType.SCALAR_ADD, name)
        return Tensor(handle, owner_op_type=OpType.SCALAR_ADD)

    def scalar_sub(self, input, scalar, inplace=True, name=None):
        """Scalar subtraction of a scalar to each entry of a tensor.

        :param input: the input Tensor.
        :type input: Tensor

        :param input: the scalar
        :type scalar: float

        :param name: the name of the layer. Default is None.
        :type name: string

        :returns:  Tensor -- the output tensor.
        """
        c_name = get_c_name(name)
        handle = ffc().flexflow_model_add_scalar_sub(
            self.handle, input.handle, scalar, inplace, c_name
        )
        self.add_layer(OpType.SCALAR_SUB, name)
        return Tensor(handle, owner_op_type=OpType.SCALAR_SUB)

    def scalar_true_divide(self, input, scalar, inplace=True, name=None):
        """Scalar regular division of a tensor by an scalar.

        :param input: the input Tensor.
        :type input: Tensor

        :param input: the scalar
        :type scalar: float

        :param name: the name of the layer. Default is None.
        :type name: string

        :returns:  Tensor -- the output tensor.
        """
        c_name = get_c_name(name)
        handle = ffc().flexflow_model_add_scalar_truediv(
            self.handle, input.handle, scalar, inplace, c_name
        )
        self.add_layer(OpType.SCALAR_TRUEDIV, name)
        return Tensor(handle, owner_op_type=OpType.SCALAR_TRUEDIV)

    def gelu(self, input, inplace=True, name=None):
        """Gaussian Error Linear Unit activation function.

        :param input: the input Tensor.
        :type input: Tensor

        :param name: the name of the layer. Default is None.
        :type name: string

        :returns:  Tensor -- the output tensor.
        """
        c_name = get_c_name(name)
        handle = ffc().flexflow_model_add_gelu(self.handle, input.handle, c_name)
        self.add_layer(OpType.GELU, name)
        return Tensor(handle, owner_op_type=OpType.GELU)

    def relu(self, input, inplace=True, name=None):
        """Rectified Linear Unit activation function.

        :param input: the input Tensor.
        :type input: Tensor

        :param name: the name of the layer. Default is None.
        :type name: string

        :returns:  Tensor -- the output tensor.
        """
        c_name = get_c_name(name)
        handle = ffc().flexflow_model_add_relu(
            self.handle, input.handle, inplace, c_name
        )
        self.add_layer(OpType.RELU, name)
        return Tensor(handle, owner_op_type=OpType.RELU)

    def identity(self, input, name=None):
        """Identity function.

        :param input: the input Tensor.
        :type input: Tensor

        :param name: the name of the layer. Default is None.
        :type name: string

        :returns:  Tensor -- the output tensor.
        """
        c_name = get_c_name(name)
        handle = ffc().flexflow_model_add_identity(self.handle, input.handle, c_name)
        self.add_layer(OpType.IDENTITY, name)
        return Tensor(handle, owner_op_type=OpType.IDENTITY)

    def sigmoid(self, input, name=None):
        """Sigmoid activation function, :math:`sigmoid(x) = 1 / (1 + exp(-x))`.

        :param input: the input Tensor.
        :type input: Tensor

        :param name: the name of the layer. Default is None.
        :type name: string

        :returns:  Tensor -- the output tensor.
        """
        c_name = get_c_name(name)
        handle = ffc().flexflow_model_add_sigmoid(self.handle, input.handle, c_name)
        self.add_layer(OpType.SIGMOID, name)
        return Tensor(handle, owner_op_type=OpType.SIGMOID)

    def tanh(self, input, name=None):
        """Hyperbolic tangent activation function.

        :param input: the input Tensor.
        :type input: Tensor

        :param name: the name of the layer. Default is None.
        :type name: string

        :returns:  Tensor -- the output tensor.
        """
        c_name = get_c_name(name)
        handle = ffc().flexflow_model_add_tanh(self.handle, input.handle, c_name)
        self.add_layer(OpType.TANH, name)
        return Tensor(handle, owner_op_type=OpType.TANH)

    def elu(self, input, inplace=True, name=None):
        """Exponential Linear Unit. activation function.

        :param input: the input Tensor.
        :type input: Tensor

        :param name: the name of the layer. Default is None.
        :type name: string

        :returns:  Tensor -- the output tensor.
        """
        c_name = get_c_name(name)
        handle = ffc().flexflow_model_add_elu(
            self.handle, input.handle, inplace, c_name
        )
        self.add_layer(OpType.ELU, name)
        return Tensor(handle, owner_op_type=OpType.ELU)

    def dropout(self, input, rate, seed, name=None):
        """The Dropout layer randomly sets input units to 0 with
        a frequency of :attr:`rate` at each step during training time,
        which helps prevent overfitting.
        Inputs not set to 0 are scaled up by 1/(1 - rate) such that the
        sum over all inputs is unchanged.

        :param input: the input Tensor.
        :type input: Tensor

        :param rate: Fraction of the input units to drop.
        :type rate: float(0-1)

        :param seed: random seed.
        :type seed: int

        :param name: the name of the layer. Default is None.
        :type name: string

        :returns:  Tensor -- the output tensor.
        """
        c_name = get_c_name(name)
        handle = ffc().flexflow_model_add_dropout(
            self.handle, input.handle, rate, seed, c_name
        )
        self.add_layer(OpType.DROPOUT, name)
        return Tensor(handle, owner_op_type=OpType.DROPOUT)

    def multihead_attention(
        self,
        query,
        key,
        value,
        embed_dim,
        num_heads,
        kdim=0,
        vdim=0,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        kernel_initializer=None,
        name=None,
    ):
        """Defines the MultiHead Attention operation as described in Attention Is All You Need
        which takes in the tensors :attr:`query`, :attr:`key`, and :attr:`value`,
        and returns the dot-product attention between them:.

        :param query: the query Tensor.
        :type query: Tensor

        :param key: the key Tensor.
        :type key: Tensor

        :param value: the value Tensor.
        :type value: Tensor

        :param embed_dim: total dimension of the model
        :type embed_dim: int

        :param num_heads: Number of attention heads.
        :type num_heads: int

        :param kdim: total number of features in key. Default is 0
        :type kdim: int

        :param vdim: total number of features in value. Default is 0
        :type vdim: int

        :param dropout: a Dropout layer on attn_output_weights. Default is 0.0
        :type dropout: float(0-1)

        :param bias: Whether the dense layers use bias vectors. Default is True.
        :type bias: bool

        :param add_bias_kv: add bias to the key and value sequences at dim=0. Default is False.
        :type add_bias_kv: bool

        :param add_zero_attn: add a new batch of zeros to the key and value sequences at dim=1. Default is False.
        :type add_zero_attn: bool

        :param kernel_initializer: Initializer for dense layer kernels. If it is set to None, the GlorotUniformInitializer is applied.
        :type kernel_initializer: Initializer

        :param name: the name of the layer. Default is None.
        :type name: string

        :returns:  Tensor -- the output tensor.
        """
        c_name = get_c_name(name)
        kernel_init_handle = self.__get_initializer_handle(kernel_initializer)
        handle = ffc().flexflow_model_add_multihead_attention(
            self.handle,
            query.handle,
            key.handle,
            value.handle,
            embed_dim,
            num_heads,
            kdim,
            vdim,
            dropout,
            bias,
            add_bias_kv,
            add_zero_attn,
            kernel_init_handle,
            c_name,
        )
        self.add_layer(OpType.MULTIHEAD_ATTENTION, name)
        return Tensor(handle, owner_op_type=OpType.MULTIHEAD_ATTENTION)

    def inc_multihead_self_attention(
        self,
        input,
        embed_dim,
        num_heads,
        kdim=0,
        vdim=0,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        data_type=DataType.DT_NONE,
        kernel_initializer=None,
        apply_rotary_embedding=False,
        scaling_query=False,
        scaling_factor=1.0,
        qk_prod_scaling=True,
        position_bias=False,
        name=None,
    ):
        """Defines the MultiHead Attention operation as described in Attention Is All You Need
        which takes in the tensors :attr:`input`, and uses it for all three of query, key and values.
        In inference mode, the attention is computed using incremental decoding.

        :param input: the input Tensor.
        :type input: Tensor

        :param embed_dim: total dimension of the model
        :type embed_dim: int

        :param num_heads: Number of attention heads.
        :type num_heads: int

        :param kdim: total number of features in key. Default is 0
        :type kdim: int

        :param vdim: total number of features in value. Default is 0
        :type vdim: int

        :param dropout: a Dropout layer on attn_output_weights. Default is 0.0
        :type dropout: float(0-1)

        :param bias: Whether the dense layers use bias vectors. Default is True.
        :type bias: bool

        :param add_bias_kv: add bias to the key and value sequences at dim=0. Default is False.
        :type add_bias_kv: bool

        :param add_zero_attn: add a new batch of zeros to the key and value sequences at dim=1. Default is False.
        :type add_zero_attn: bool

        :param data_type: the data type of the tensors. Default is DataType.DT_NONE, which means using the data type of the input tensors.
        :type data_type: DataType

        :param kernel_initializer: Initializer for dense layer kernels. If it is set to None, the GlorotUniformInitializer is applied.
        :type kernel_initializer: Initializer

        :param apply_rotary_embedding: Whether to apply rotary embeddings. Default is False.
        :type apply_rotary_embedding: bool

        :param scaling_query: Whether to apply scaling query. Default is False.
        :type scaling_query: bool

        :param scaling_factor: The scaling factor to use for scaling. Default is 1.0.
        :type scaling_factor: float

        :param qk_prod_scaling: Whether to apply scaling to the QK product. Default is True.
        :type qk_prod_scaling: bool

        :param position_bias: Whether to add position bias to the QK product. Default is False.
        :type position_bias: bool

        :param name: the name of the layer. Default is None.
        :type name: string

        :returns:  Tensor -- the output tensor.
        """
        c_name = get_c_name(name)
        kernel_init_handle = self.__get_initializer_handle(kernel_initializer)
        c_data_type = enum_to_int(DataType, data_type)
        handle = ffc().flexflow_model_add_inc_multihead_self_attention(
            self.handle,
            input.handle,
            embed_dim,
            num_heads,
            kdim,
            vdim,
            dropout,
            bias,
            add_bias_kv,
            add_zero_attn,
            c_data_type,
            kernel_init_handle,
            apply_rotary_embedding,
            scaling_query,
            scaling_factor,
            qk_prod_scaling,
            position_bias,
            c_name,
        )
        self.add_layer(OpType.INC_MULTIHEAD_ATTENTION, name)
        return Tensor(handle, owner_op_type=OpType.INC_MULTIHEAD_ATTENTION)

    def spec_inc_multihead_self_attention(
        self,
        input,
        embed_dim,
        num_heads,
        kdim=0,
        vdim=0,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        data_type=DataType.DT_NONE,
        kernel_initializer=None,
        apply_rotary_embedding=False,
        scaling_query=False,
        scaling_factor=1.0,
        qk_prod_scaling=True,
        position_bias=False,
        name=None,
    ):
        """Defines the MultiHead Attention operation as described in Attention Is All You Need
        which takes in the tensors :attr:`input`, and uses it for all three of query, key and values.
        This operator only supports computing the attention in inference (beam search) mode.

        :param input: the input Tensor.
        :type input: Tensor

        :param embed_dim: total dimension of the model
        :type embed_dim: int

        :param num_heads: Number of attention heads.
        :type num_heads: int

        :param kdim: total number of features in key. Default is 0
        :type kdim: int

        :param vdim: total number of features in value. Default is 0
        :type vdim: int

        :param dropout: a Dropout layer on attn_output_weights. Default is 0.0
        :type dropout: float(0-1)

        :param bias: Whether the dense layers use bias vectors. Default is True.
        :type bias: bool

        :param add_bias_kv: add bias to the key and value sequences at dim=0. Default is False.
        :type add_bias_kv: bool

        :param add_zero_attn: add a new batch of zeros to the key and value sequences at dim=1. Default is False.
        :type add_zero_attn: bool

        :param data_type: the data type of the tensors. Default is DataType.DT_NONE, which means using the data type of the input tensors.
        :type data_type: DataType

        :param kernel_initializer: Initializer for dense layer kernels. If it is set to None, the GlorotUniformInitializer is applied.
        :type kernel_initializer: Initializer

        :param apply_rotary_embedding: Whether to apply rotary embeddings. Default is False.
        :type apply_rotary_embedding: bool

        :param scaling_query: Whether to apply scaling query. Default is False.
        :type scaling_query: bool

        :param scaling_factor: The scaling factor to use for scaling. Default is 1.0.
        :type scaling_factor: float

        :param qk_prod_scaling: Whether to apply scaling to the QK product. Default is True.
        :type qk_prod_scaling: bool

        :param position_bias: Whether to add position bias to the QK product. Default is False.
        :type position_bias: bool

        :param name: the name of the layer. Default is None.
        :type name: string

        :returns:  Tensor -- the output tensor.
        """
        c_name = get_c_name(name)
        kernel_init_handle = self.__get_initializer_handle(kernel_initializer)
        c_data_type = enum_to_int(DataType, data_type)
        handle = ffc().flexflow_model_add_spec_inc_multihead_self_attention(
            self.handle,
            input.handle,
            embed_dim,
            num_heads,
            kdim,
            vdim,
            dropout,
            bias,
            add_bias_kv,
            add_zero_attn,
            c_data_type,
            kernel_init_handle,
            apply_rotary_embedding,
            scaling_query,
            scaling_factor,
            qk_prod_scaling,
            position_bias,
            c_name,
        )
        self.add_layer(OpType.SPEC_INC_MULTIHEAD_SELF_ATTENTION, name)
        return Tensor(handle, owner_op_type=OpType.SPEC_INC_MULTIHEAD_SELF_ATTENTION)

    def inc_multihead_self_attention_verify(
        self,
        input,
        embed_dim,
        num_heads,
        kdim=0,
        vdim=0,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        data_type=DataType.DT_NONE,
        kernel_initializer=None,
        apply_rotary_embedding=False,
        scaling_query=False,
        scaling_factor=1.0,
        qk_prod_scaling=True,
        position_bias=False,
        name=None,
    ):
        """Defines the MultiHead Attention operation as described in Attention Is All You Need
        which takes in the tensors :attr:`input`, and uses it for all three of query, key and values.
        This operator only supports computing the attention in inference (tree verify) mode.

        :param input: the input Tensor.
        :type input: Tensor

        :param embed_dim: total dimension of the model
        :type embed_dim: int

        :param num_heads: Number of attention heads.
        :type num_heads: int

        :param kdim: total number of features in key. Default is 0
        :type kdim: int

        :param vdim: total number of features in value. Default is 0
        :type vdim: int

        :param dropout: a Dropout layer on attn_output_weights. Default is 0.0
        :type dropout: float(0-1)

        :param bias: Whether the dense layers use bias vectors. Default is True.
        :type bias: bool

        :param add_bias_kv: add bias to the key and value sequences at dim=0. Default is False.
        :type add_bias_kv: bool

        :param add_zero_attn: add a new batch of zeros to the key and value sequences at dim=1. Default is False.
        :type add_zero_attn: bool

        :param data_type: the data type of the tensors. Default is DataType.DT_NONE, which means using the data type of the input tensors.
        :type data_type: DataType

        :param kernel_initializer: Initializer for dense layer kernels. If it is set to None, the GlorotUniformInitializer is applied.
        :type kernel_initializer: Initializer

        :param apply_rotary_embedding: Whether to apply rotary embeddings. Default is False.
        :type apply_rotary_embedding: bool

        :param scaling_query: Whether to apply scaling query. Default is False.
        :type scaling_query: bool

        :param scaling_factor: The scaling factor to use for scaling. Default is 1.0.
        :type scaling_factor: float

        :param qk_prod_scaling: Whether to apply scaling to the QK product. Default is True.
        :type qk_prod_scaling: bool

        :param position_bias: Whether to add position bias to the QK product. Default is False.
        :type position_bias: bool

        :param name: the name of the layer. Default is None.
        :type name: string

        :returns:  Tensor -- the output tensor.
        """
        c_name = get_c_name(name)
        kernel_init_handle = self.__get_initializer_handle(kernel_initializer)
        c_data_type = enum_to_int(DataType, data_type)
        handle = ffc().flexflow_model_add_inc_multihead_self_attention_verify(
            self.handle,
            input.handle,
            embed_dim,
            num_heads,
            kdim,
            vdim,
            dropout,
            bias,
            add_bias_kv,
            add_zero_attn,
            c_data_type,
            kernel_init_handle,
            apply_rotary_embedding,
            scaling_query,
            scaling_factor,
            qk_prod_scaling,
            position_bias,
            c_name,
        )
        self.add_layer(OpType.TREE_INC_MULTIHEAD_SELF_ATTENTION, name)
        return Tensor(handle, owner_op_type=OpType.TREE_INC_MULTIHEAD_SELF_ATTENTION)

    def inc_multiquery_self_attention(
        self,
        input,
        embed_dim,
        num_q_heads,
        num_kv_heads,
        kdim=0,
        vdim=0,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        data_type=DataType.DT_NONE,
        kernel_initializer=None,
        apply_rotary_embedding=False,
        scaling_query=False,
        scaling_factor=1.0,
        qk_prod_scaling=True,
        position_bias=False,
        name=None,
    ):
        """Defines the multi-query head attention, which allows a different number of Q and KV heads,
        and takes in the tensors :attr:`input`, and uses it for all three of query, key and values.
        In inference mode, the attention is computed using incremental decoding.

        :param input: the input Tensor.
        :type input: Tensor

        :param embed_dim: total dimension of the model
        :type embed_dim: int

        :param num_q_heads: Number of query attention heads.
        :type num_q_heads: int

        :param num_kv_heads: Number of key/value attention heads.
        :type num_kv_heads: int

        :param kdim: total number of features in key. Default is 0
        :type kdim: int

        :param vdim: total number of features in value. Default is 0
        :type vdim: int

        :param dropout: a Dropout layer on attn_output_weights. Default is 0.0
        :type dropout: float(0-1)

        :param bias: Whether the dense layers use bias vectors. Default is True.
        :type bias: bool

        :param add_bias_kv: add bias to the key and value sequences at dim=0. Default is False.
        :type add_bias_kv: bool

        :param add_zero_attn: add a new batch of zeros to the key and value sequences at dim=1. Default is False.
        :type add_zero_attn: bool

        :param data_type: the data type of the tensors. Default is DataType.DT_NONE, which means using the data type of the input tensors.
        :type data_type: DataType

        :param kernel_initializer: Initializer for dense layer kernels. If it is set to None, the GlorotUniformInitializer is applied.
        :type kernel_initializer: Initializer

        :param apply_rotary_embedding: Whether to apply rotary embeddings. Default is False.
        :type apply_rotary_embedding: bool

        :param scaling_query: Whether to apply scaling query. Default is False.
        :type scaling_query: bool

        :param scaling_factor: The scaling factor to use for scaling. Default is 1.0.
        :type scaling_factor: float

        :param qk_prod_scaling: Whether to apply scaling to the QK product. Default is True.
        :type qk_prod_scaling: bool

        :param position_bias: Whether to add position bias to the QK product. Default is False.
        :type position_bias: bool

        :param name: the name of the layer. Default is None.
        :type name: string

        :returns:  Tensor -- the output tensor.
        """
        c_name = get_c_name(name)
        kernel_init_handle = self.__get_initializer_handle(kernel_initializer)
        c_data_type = enum_to_int(DataType, data_type)
        handle = ffc().flexflow_model_add_inc_multiquery_self_attention(
            self.handle,
            input.handle,
            embed_dim,
            num_q_heads,
            num_kv_heads,
            kdim,
            vdim,
            dropout,
            bias,
            add_bias_kv,
            add_zero_attn,
            c_data_type,
            kernel_init_handle,
            apply_rotary_embedding,
            scaling_query,
            scaling_factor,
            qk_prod_scaling,
            position_bias,
            c_name,
        )
        self.add_layer(OpType.INC_MULTIHEAD_ATTENTION, name)
        return Tensor(handle, owner_op_type=OpType.INC_MULTIHEAD_ATTENTION)

    def spec_inc_multiquery_self_attention(
        self,
        input,
        embed_dim,
        num_q_heads,
        num_kv_heads,
        kdim=0,
        vdim=0,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        data_type=DataType.DT_NONE,
        kernel_initializer=None,
        apply_rotary_embedding=False,
        scaling_query=False,
        scaling_factor=1.0,
        qk_prod_scaling=True,
        position_bias=False,
        name=None,
    ):
        """Defines the multi-query head attention, which allows a different number of Q and KV heads,
        and takes in the tensors :attr:`input`, and uses it for all three of query, key and values.
        This operator only supports computing the attention in inference (beam search) mode.

        :param input: the input Tensor.
        :type input: Tensor

        :param embed_dim: total dimension of the model
        :type embed_dim: int

        :param num_q_heads: Number of query attention heads.
        :type num_q_heads: int

        :param num_kv_heads: Number of key/value attention heads.
        :type num_kv_heads: int

        :param kdim: total number of features in key. Default is 0
        :type kdim: int

        :param vdim: total number of features in value. Default is 0
        :type vdim: int

        :param dropout: a Dropout layer on attn_output_weights. Default is 0.0
        :type dropout: float(0-1)

        :param bias: Whether the dense layers use bias vectors. Default is True.
        :type bias: bool

        :param add_bias_kv: add bias to the key and value sequences at dim=0. Default is False.
        :type add_bias_kv: bool

        :param add_zero_attn: add a new batch of zeros to the key and value sequences at dim=1. Default is False.
        :type add_zero_attn: bool

        :param data_type: the data type of the tensors. Default is DataType.DT_NONE, which means using the data type of the input tensors.
        :type data_type: DataType

        :param kernel_initializer: Initializer for dense layer kernels. If it is set to None, the GlorotUniformInitializer is applied.
        :type kernel_initializer: Initializer

        :param apply_rotary_embedding: Whether to apply rotary embeddings. Default is False.
        :type apply_rotary_embedding: bool

        :param scaling_query: Whether to apply scaling query. Default is False.
        :type scaling_query: bool

        :param scaling_factor: The scaling factor to use for scaling. Default is 1.0.
        :type scaling_factor: float

        :param qk_prod_scaling: Whether to apply scaling to the QK product. Default is True.
        :type qk_prod_scaling: bool

        :param position_bias: Whether to add position bias to the QK product. Default is False.
        :type position_bias: bool

        :param name: the name of the layer. Default is None.
        :type name: string

        :returns:  Tensor -- the output tensor.
        """
        c_name = get_c_name(name)
        kernel_init_handle = self.__get_initializer_handle(kernel_initializer)
        c_data_type = enum_to_int(DataType, data_type)
        handle = ffc().flexflow_model_add_spec_inc_multiquery_self_attention(
            self.handle,
            input.handle,
            embed_dim,
            num_q_heads,
            num_kv_heads,
            kdim,
            vdim,
            dropout,
            bias,
            add_bias_kv,
            add_zero_attn,
            c_data_type,
            kernel_init_handle,
            apply_rotary_embedding,
            scaling_query,
            scaling_factor,
            qk_prod_scaling,
            position_bias,
            c_name,
        )
        self.add_layer(OpType.SPEC_INC_MULTIHEAD_SELF_ATTENTION, name)
        return Tensor(handle, owner_op_type=OpType.SPEC_INC_MULTIHEAD_SELF_ATTENTION)

    def inc_multiquery_self_attention_verify(
        self,
        input,
        embed_dim,
        num_q_heads,
        num_kv_heads,
        kdim=0,
        vdim=0,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        data_type=DataType.DT_NONE,
        kernel_initializer=None,
        apply_rotary_embedding=False,
        scaling_query=False,
        scaling_factor=1.0,
        qk_prod_scaling=True,
        position_bias=False,
        name=None,
    ):
        """Defines the multi-query head attention, which allows a different number of Q and KV heads,
        and takes in the tensors :attr:`input`, and uses it for all three of query, key and values.
        This operator only supports computing the attention in inference (tree verify) mode.

        :param input: the input Tensor.
        :type input: Tensor

        :param embed_dim: total dimension of the model
        :type embed_dim: int

        :param num_q_heads: Number of query attention heads.
        :type num_q_heads: int

        :param num_kv_heads: Number of key/value attention heads.
        :type num_kv_heads: int

        :param kdim: total number of features in key. Default is 0
        :type kdim: int

        :param vdim: total number of features in value. Default is 0
        :type vdim: int

        :param dropout: a Dropout layer on attn_output_weights. Default is 0.0
        :type dropout: float(0-1)

        :param bias: Whether the dense layers use bias vectors. Default is True.
        :type bias: bool

        :param add_bias_kv: add bias to the key and value sequences at dim=0. Default is False.
        :type add_bias_kv: bool

        :param add_zero_attn: add a new batch of zeros to the key and value sequences at dim=1. Default is False.
        :type add_zero_attn: bool

        :param data_type: the data type of the tensors. Default is DataType.DT_NONE, which means using the data type of the input tensors.
        :type data_type: DataType

        :param kernel_initializer: Initializer for dense layer kernels. If it is set to None, the GlorotUniformInitializer is applied.
        :type kernel_initializer: Initializer

        :param apply_rotary_embedding: Whether to apply rotary embeddings. Default is False.
        :type apply_rotary_embedding: bool

        :param scaling_query: Whether to apply scaling query. Default is False.
        :type scaling_query: bool

        :param scaling_factor: The scaling factor to use for scaling. Default is 1.0.
        :type scaling_factor: float

        :param qk_prod_scaling: Whether to apply scaling to the QK product. Default is True.
        :type qk_prod_scaling: bool

        :param position_bias: Whether to add position bias to the QK product. Default is False.
        :type position_bias: bool

        :param name: the name of the layer. Default is None.
        :type name: string

        :returns:  Tensor -- the output tensor.
        """
        c_name = get_c_name(name)
        kernel_init_handle = self.__get_initializer_handle(kernel_initializer)
        c_data_type = enum_to_int(DataType, data_type)
        handle = ffc().flexflow_model_add_inc_multiquery_self_attention_verify(
            self.handle,
            input.handle,
            embed_dim,
            num_q_heads,
            num_kv_heads,
            kdim,
            vdim,
            dropout,
            bias,
            add_bias_kv,
            add_zero_attn,
            c_data_type,
            kernel_init_handle,
            apply_rotary_embedding,
            scaling_query,
            scaling_factor,
            qk_prod_scaling,
            position_bias,
            c_name,
        )
        self.add_layer(OpType.TREE_INC_MULTIHEAD_SELF_ATTENTION, name)
        return Tensor(handle, owner_op_type=OpType.TREE_INC_MULTIHEAD_SELF_ATTENTION)

    def rms_norm(self, input, eps, dim, name=None):
        """Defines the RMS Norm layer.

        :param input: the input Tensor.
        :type input: Tensor

        :param eps: a value added to the denominator for numerical stability
        :type eps: float

        :param dim: The dimension with respect to which to take the norm
        :type dim: int

        :param name: the name of the layer. Default is None.
        :type name: string

        :returns:  Tensor -- the output tensor.
        """
        c_name = get_c_name(name)
        handle = ffc().flexflow_model_add_rms_norm(
            self.handle, input.handle, eps, dim, c_name
        )
        self.add_layer(OpType.RMS_NORM, name)
        return Tensor(handle, owner_op_type=OpType.RMS_NORM)

    def residual_rms_norm(self, input1, input2, eps, dim, inplace_residual=False, name=None):
        """Defines the Residual RMS Norm layer.

        :param input: the input 1 Tensor.
        :type input: Tensor

        :param input: the input 2 Tensor.
        :type input: Tensor

        :param eps: a value added to the denominator for numerical stability
        :type eps: float

        :param dim: The dimension with respect to which to take the norm
        :type dim: int

        :param name: the name of the layer. Default is None.
        :type name: string

        :param inplace_residual: whether to compute the residual inplace using the input tensor. Default is False.
        :type inplace_residual: bool

        :returns:  Tensor -- the output tensor.
        """
        c_name = get_c_name(name)
        handles_array = ffc().flexflow_model_add_residual_rms_norm(
            self.handle, input1.handle, input2.handle, eps, dim, inplace_residual, c_name
        )
        self.add_layer(OpType.RESIDUAL_RMS_NORM, name)
        return Tensor(handles_array[0], owner_op_type=OpType.RESIDUAL_RMS_NORM), Tensor(
            handles_array[1], owner_op_type=OpType.RESIDUAL_RMS_NORM
        )

    def arg_top_k(self, input, k, sorted, speculative_decoding, name=None):
        """Defines the Arg TopK layer.

        :param input: the input Tensor.
        :type input: Tensor

        :param k: the top k indices to select
        :type k: int

        :param sorted: Whether the entries should be sorted
        :type sorted: bool

        :param speculative_decoding: Whether you need to perform beam search
        :type speculative_decoding: bool

        :param name: the name of the layer. Default is None.
        :type name: string

        :returns:  Tensor -- the output tensor.
        """
        c_name = get_c_name(name)
        handle = ffc().flexflow_model_add_arg_top_k(
            self.handle, input.handle, k, sorted, c_name
        )
        self.add_layer(OpType.ARG_TOPK, name)
        return Tensor(handle, owner_op_type=OpType.ARG_TOPK)

    def beam_top_k(self, input, max_beam_size, sorted, name=None):
        """Defines the Beam TopK layer.

        :param input: the input Tensor.
        :type input: Tensor

        :param max_beam_size: the top max_beam_size indices to select
        :type max_beam_size: int

        :param sorted: Whether the entries should be sorted
        :type sorted: bool

        :param name: the name of the layer. Default is None.
        :type name: string

        :returns:  Tensor -- the output tensor.
        """
        c_name = get_c_name(name)
        handle = ffc().flexflow_model_add_beam_top_k(
            self.handle, input.handle, max_beam_size, sorted, c_name
        )
        self.add_layer(OpType.BEAM_TOPK, name)
        return Tensor(handle, owner_op_type=OpType.BEAM_TOPK)

    def sampling(self, input, top_p, name=None):
        """Defines the Sampling layer.

        :param input: the input Tensor.
        :type input: Tensor

        :param top_p: The top_p parameter of the sampling
        :type top_p: float

        :param name: the name of the layer. Default is None.
        :type name: string

        :returns:  Tensor -- the output tensor.
        """
        c_name = get_c_name(name)
        handle = ffc().flexflow_model_add_sampling(
            self.handle, input.handle, top_p, c_name
        )
        self.add_layer(OpType.SAMPLING, name)
        return Tensor(handle, owner_op_type=OpType.SAMPLING)

    def argmax(self, input, beam_search, name=None):
        """Defines the Sampling layer.

        :param input: the input Tensor.
        :type input: Tensor

        :param beam_search: Whether you need to perform beam search
        :type beam_search: bool

        :param name: the name of the layer. Default is None.
        :type name: string

        :returns:  Tensor -- the output tensor.
        """
        c_name = get_c_name(name)
        handle = ffc().flexflow_model_add_argmax(
            self.handle, input.handle, beam_search, c_name
        )
        self.add_layer(OpType.ARGMAX, name)
        return Tensor(handle, owner_op_type=OpType.ARGMAX)

    def reset_metrics(self):
        """Reset performance metrics.

        :returns:  None -- no returns.
        """
        ffc().flexflow_model_reset_metrics(self.handle)

    def init_layers(self):
        """Initialize layers.

        :returns:  None -- no returns.
        """
        ffc().flexflow_model_init_layers(self.handle)

    def prefetch(self):
        ffc().flexflow_model_prefetch(self.handle)

    def forward(self, seq_length=None):
        """Forward propagation of all layers.

        :returns:  None -- no returns.
        """
        if seq_length is None:
            seq_length = -1
        ffc().flexflow_model_forward(self.handle, seq_length)

    # TODO: seperate compute_metrics from backward
    def backward(self, seq_length=None):
        """Backward propagation of all layers.

        :returns:  None -- no returns.
        """
        if seq_length is None:
            seq_length = -1
        ffc().flexflow_model_backward(self.handle, seq_length)

    def compute_metrics(self):
        """Compute performance metrics.

        :returns:  None -- no returns.
        """
        ffc().flexflow_model_compute_metrics(self.handle)

    def update(self):
        """Update weights and biases of all layers.

        :returns:  None -- no returns.
        """
        ffc().flexflow_model_update(self.handle)

    def compile(self, optimizer=None, loss_type=None, metrics=None, comp_mode=None):
        """Configure the model for trainting. FlexFlow uses lazy initialization,
        so the actual creating of all operations (including creating and partitioning
        of weight, bias and output tensors) happen during compile.

        :param optimizer: optimizer instance.
        :type optimizer: Optimizer

        :param loss_type: Enum of LossType.
          Options are LOSS_CATEGORICAL_CROSSENTROPY, LOSS_SPARSE_CATEGORICAL_CROSSENTROPY,
          LOSS_MEAN_SQUARED_ERROR_AVG_REDUCE and LOSS_MEAN_SQUARED_ERROR_SUM_REDUCE.
        :type loss_type: LossType

        :param metrics: List of metrics to be evaluated by the model during training and testing.
          Each of this is a Enum of MetricsType. Options are METRICS_ACCURACY,
          METRICS_CATEGORICAL_CROSSENTROPY, METRICS_SPARSE_CATEGORICAL_CROSSENTROPY,
          METRICS_MEAN_SQUARED_ERROR, METRICS_ROOT_MEAN_SQUARED_ERROR, METRICS_MEAN_ABSOLUTE_ERROR
        :type metrics: MetricsType

        :param comp_mode: Enum of CompMode.
          Options are COMP_MODE_TRAINING, COMP_MODE_INFERENCE
        :type comp_mode: CompMode

        :returns:  None -- no returns.
        """
        self.optimizer = optimizer

        c_loss_type = enum_to_int(LossType, loss_type)
        metrics_int = []
        for metric in metrics:
            metrics_int.append(enum_to_int(MetricsType, metric))
        c_metrics = ffi.new("int[]", metrics_int)
        if comp_mode == None:
            comp_mode = CompMode.TRAINING
        c_comp_mode = enum_to_int(CompMode, comp_mode)
        ffc().flexflow_model_compile(
            self.handle, c_loss_type, c_metrics, len(metrics), c_comp_mode
        )
        for ff_tensor, np_tensor in self.attr_tensors.items():
            ff_tensor.set_tensor(self, np_tensor)
        print("Compiled ffmodel!")

    def fit(self, x=None, y=None, batch_size=None, epochs=1):
        """Trains the model for a fixed number of epochs (iterations on a dataset).

        :param x: Input data. It can be a Dataloader instance or a list of Dataloader instances.
        :type x: Dataloader

        :param y: Target data (label). It can be a Dataloader instance or a list of Dataloader instances.
        :type y: Dataloader

        :param batch_size: Number of samples per gradient update. It must be identical with :attr:`-b`
          or :attr:`--batch-size` from the command line.
        :type batch_size: int

        :param epochs: Number of epochs to train the model.
          An epoch is an iteration over the entire :attr:`x` and :attr:`y` data provided.
          The default value is 1.
        :type epochs: int

        :returns:  None -- no returns.
        """
        if isinstance(x, list) == False:
            dataloaders = [x]
        else:
            dataloaders = x
        dataloaders.append(y)

        num_samples = y.num_samples
        batch_size = self._ffconfig.batch_size
        self._tracing_id += 1  # get a new tracing id
        for epoch in range(0, epochs):
            for d in dataloaders:
                d.reset()
            self.reset_metrics()
            iterations = num_samples / batch_size
            for iter in range(0, int(iterations)):
                self._ffconfig.begin_trace(self._tracing_id)
                for d in dataloaders:
                    d.next_batch(self)
                self.forward()
                self.zero_gradients()
                self.backward()
                self.update()
                self._ffconfig.end_trace(self._tracing_id)

    def eval(self, x=None, y=None, batch_size=None):
        """Returns the loss value & metrics values for the model in test mode.

        :param x: Input data. It can be a Dataloader instance or a list of Dataloader instances.
        :type x: Dataloader

        :param y: Target data (label). It can be a Dataloader instance or a list of Dataloader instances.
        :type y: Dataloader

        :param batch_size: Number of samples per gradient update. It must be identical with :attr:`-b`
          or :attr:`--batch-size` from the command line.
        :type batch_size: int

        :param epochs: Number of epochs to train the model.
          An epoch is an iteration over the entire :attr:`x` and :attr:`y` data provided.
          The default value is 1.
        :type epochs: int

        :returns:  None -- no returns.
        """
        if isinstance(x, list) == False:
            dataloaders = [x]
        else:
            dataloaders = x
        dataloaders.append(y)

        num_samples = y.num_samples
        batch_size = self._ffconfig.batch_size
        for d in dataloaders:
            d.reset()
        self.reset_metrics()
        iterations = num_samples / batch_size
        self._tracing_id += 1  # get a new tracing id
        for iter in range(0, int(iterations)):
            for d in dataloaders:
                d.next_batch(self)
            self._ffconfig.begin_trace(self._tracing_id)
            self.forward()
            self.compute_metrics()
            self._ffconfig.end_trace(self._tracing_id)

    def zero_gradients(self):
        """Empty the gradients of all layers.

        :returns:  None -- no returns.
        """
        ffc().flexflow_model_zero_gradients(self.handle)

    def set_optimizer(self, optimizer):
        if isinstance(optimizer, SGDOptimizer) == True:
            ffc().flexflow_model_set_sgd_optimizer(self.handle, optimizer.handle)
        elif isinstance(optimizer, AdamOptimizer) == True:
            ffc().flexflow_model_set_adam_optimizer(self.handle, optimizer.handle)
        elif optimizer == None:
            pass
        else:
            assert 0, "[Model]: unknown optimizer"

    optimizer = property(fset=set_optimizer)

    def print_layers(self, id=-1):
        ffc().flexflow_model_print_layers(self.handle, id)

    def get_layer_by_id(self, layer_id):
        return self._layers[layer_id]

    def get_last_layer(self):
        return self._layers[self._nb_layers - 1]

    def get_layer_by_name(self, layer_name):
        for layer_id in self._layers:
            layer = self._layers[layer_id]
            if layer.name == layer_name:
                return layer
        assert 0, f"Cannot find the layer with name {layer_name}"
        return None

    def get_tensor_by_id(self, id):
        handle = ffc().flexflow_model_get_parameter_by_id(self.handle, id)
        return Parameter(handle)

    @property
    def label_tensor(self):
        handle = ffc().flexflow_model_get_label_tensor(self.handle)
        return Tensor(handle, deallocate=False)

    def get_perf_metrics(self):
        handle = ffc().flexflow_model_get_perf_metrics(self.handle)
        return PerfMetrics(handle)

    def set_transformer_layer_id(self, id):
        ffc().flexflow_model_set_transformer_layer_id(self.handle, id)

    def create_data_loader(self, batch_tensor, full_array):
        """Create a SingleDataloader instance.

        :param batch_tensor: a batch-sized tensor. Usually it is a input tensor of the model.
        :type batch_tensor: Tensor

        :param full_array: the entire data.
        :type full_array: Numpy Array

        :returns:  SingleDataloader -- returns a dataloader instance.
        """

        if self._ffconfig.enable_control_replication:
            assert (
                self._ffconfig.python_data_loader_type != 1
            ), "To enable control replication, please set --python-data-loader-type 2"
            return self.__create_data_loader_ptr(batch_tensor, full_array)
        else:
            if self._ffconfig.python_data_loader_type == 1:
                return self.__create_data_loader_attach(batch_tensor, full_array)
            else:
                return self.__create_data_loader_ptr(batch_tensor, full_array)

    def __create_data_loader_attach(self, batch_tensor, full_array):
        full_array_shape = full_array.shape
        num_samples = full_array_shape[0]
        num_dim = len(full_array_shape)
        if full_array.dtype == "float16":
            datatype = DataType.DT_HALF
        elif full_array.dtype == "float32":
            datatype = DataType.DT_FLOAT
        elif full_array.dtype == "int32":
            datatype = DataType.DT_INT32
        elif full_array.dtype == "int64":
            datatype = DataType.DT_INT64
        else:
            assert 0, "unsupported datatype"

        if num_dim == 2:
            full_tensor = self.create_tensor(
                [num_samples, full_array_shape[1]], datatype
            )
            self.map_tensor(full_tensor)
        elif num_dim == 4:
            full_tensor = self.create_tensor(
                [
                    num_samples,
                    full_array_shape[1],
                    full_array_shape[2],
                    full_array_shape[3],
                ],
                datatype,
            )
            self.map_tensor(full_tensor)
        else:
            assert 0, "unsupported dims"

        full_tensor.attach_numpy_array(self._ffconfig, full_array)
        dataloader = SingleDataLoader(
            self, batch_tensor, full_tensor, num_samples, datatype
        )
        full_tensor.detach_numpy_array(self._ffconfig)

        return dataloader

    def __create_data_loader_ptr(self, batch_tensor, full_array):
        full_array_shape = full_array.shape
        num_samples = full_array_shape[0]
        if full_array.dtype == "float16":
            datatype = DataType.DT_HALF
        elif full_array.dtype == "float32":
            datatype = DataType.DT_FLOAT
        elif full_array.dtype == "int32":
            datatype = DataType.DT_INT32
        elif full_array.dtype == "int64":
            datatype = DataType.DT_INT64
        else:
            assert 0, "unsupported datatype"
        np_raw_ptr = full_array.__array_interface__["data"]
        raw_ptr = ffi.cast("float*", np_raw_ptr[0])
        print(
            "numpy array: %s, %s, %s"
            % (str(np_raw_ptr), str(raw_ptr), hex(np_raw_ptr[0]))
        )
        dataloader = SingleDataLoader(
            self, batch_tensor, raw_ptr, num_samples, datatype
        )

        return dataloader

    def __get_initializer_handle(self, initializer):
        if initializer == None:
            null_initializer = Initializer(None)
            return null_initializer.handle
        else:
            return initializer.handle

    def __get_op_handle(self, shared_op):
        if shared_op == None:
            op_handle = ffi.new("flexflow_op_t *")
            op_handle.impl = ffi.NULL
            op = Op(op_handle[0])
        else:
            op = shared_op
        return op.handle

    def get_output_tensor(self, ffmodel, data_type):
        shape = self.dims
        if data_type == DataType.DT_HALF:
            np_array = np.empty(shape, dtype=np.float16)
        elif data_type == DataType.DT_FLOAT:
            np_array = np.empty(shape, dtype=np.float32)
        elif self.data_type == DataType.DT_INT32:
            np_array = np.empty(shape, dtype=np.int32)
        elif self.data_type == DataType.DT_INT64:
            np_array = np.empty(shape, dtype=np.int64)
        else:
            assert 0, f"Unsupported datatype: {self.data_type}"
        np_raw_ptr = np_array.__array_interface__["data"]
        if np_array.dtype == np.float32:
            raw_ptr = ffi.cast("float*", np_raw_ptr[0])
            ret_val = ffc().flexflow_tensor_get_tensor_float(
                self.handle, ffmodel.handle, raw_ptr, False
            )
        elif np_array.dtype == np.int32:
            raw_ptr = ffi.cast("int*", np_raw_ptr[0])
            ret_val = ffc().flexflow_tensor_get_tensor_int(
                self.handle, ffmodel.handle, raw_ptr, False
            )
        elif np_array.dtype == np.int64:
            raw_ptr = ffi.cast("int64_t*", np_raw_ptr[0])
            ret_val = ffc().flexflow_tensor_get_tensor_int64(
                self.handle, ffmodel.handle, raw_ptr, False
            )
        fflogger.debug(
            "get weights raw_ptr: %s, %s, %s, %s"
            % (str(raw_ptr), str(np_raw_ptr[0]), hex(np_raw_ptr[0]), str(shape))
        )
        assert ret_val == True
        return np_array

    def generate_inf_only(self, prompt_list: List[str], max_sequence_length: int = 128):
        assert isinstance(prompt_list, list)
        c_input_texts = [get_c_name(prompt) for prompt in prompt_list]
        max_num_chars = 5 * (max_sequence_length + 100)
        c_output_texts = [ffi.new("char[]", max_num_chars) for prompt in prompt_list]
        c_output_length_and_tokens = [ffi.new("int[]", max_sequence_length + 100) for prompt in prompt_list]
        c_request_types = [enum_to_int(RequestType, RequestType.REQ_INFERENCE) for prompt in prompt_list]
        max_sequence_lengths = [max_sequence_length for prompt in prompt_list]
        peft_model_ids = [None for prompt in prompt_list]
        dataset_filepaths = [None for prompt in prompt_list]
        training_steps = [0 for prompt in prompt_list]
        ffc().flexflow_model_generate(
            self.handle,
            len(prompt_list),
            c_request_types,
            c_input_texts,
            c_output_texts,
            max_sequence_lengths,
            peft_model_ids,
            dataset_filepaths,
            training_steps,
            c_output_length_and_tokens,
        )
        from flexflow.serve import GenerationResult
        return [GenerationResult(ffi.string(c_output_text), []) for c_output_text in c_output_texts]
    
    def generate(self, requests_list: List[Request]):
        assert isinstance(requests_list, list)
        c_input_texts = [get_c_name(request.prompt) for request in requests_list] # entry will be None for finetuning requests
        c_output_texts = [ffi.new("char[]", 5 * (request.max_sequence_length + 100)) if request.req_type == RequestType.REQ_INFERENCE else ffi.NULL for request in requests_list]
        c_output_length_and_tokens = [ffi.new("int[]", request.max_sequence_length + 100) for request in requests_list]
        c_request_types = [enum_to_int(RequestType, request.req_type) for request in requests_list]
        max_sequence_lengths = [request.max_sequence_length for request in requests_list]
        peft_model_ids = [request.peft_model_id for request in requests_list]
        dataset_filepaths = [request.dataset_filepath for request in requests_list]
        training_steps = [request.max_training_steps for request in requests_list]
        ffc().flexflow_model_generate(
            self.handle,
            len(requests_list),
            c_request_types,
            c_input_texts,
            c_output_texts,
            max_sequence_lengths,
            peft_model_ids,
            dataset_filepaths,
            training_steps,
            c_output_length_and_tokens,
        )
        return [GenerationResult(ffi.string(c_output_text), []) if c_output_text != ffi.NULL else None for c_output_text in c_output_texts]

    def set_position_offset(self, offset):
        ffc().flexflow_model_set_position_offset(self.handle, offset)


# -----------------------------------------------------------------------
# SGDOptimizer
# -----------------------------------------------------------------------


class SGDOptimizer(object):
    __slots__ = ["handle", "_handle"]

    def __init__(
        self, ffmodel, lr=0.01, momentum=0.0, nesterov=False, weight_decay=0.0
    ):
        self.handle = ffc().flexflow_sgd_optimizer_create(
            ffmodel.handle, lr, momentum, nesterov, weight_decay
        )
        self._handle = ffi.gc(self.handle, ffc().flexflow_sgd_optimizer_destroy)

    def set_learning_rate(self, learning_rate):
        ffc().flexflow_sgd_optimizer_set_lr(self.handle, learning_rate)


# -----------------------------------------------------------------------
# AdamOptimizer
# -----------------------------------------------------------------------


class AdamOptimizer(object):
    __slots__ = ["handle", "_handle"]

    def __init__(
        self,
        ffmodel,
        alpha=0.001,
        beta1=0.9,
        beta2=0.999,
        weight_decay=0.0,
        epsilon=1e-8,
    ):
        self.handle = ffc().flexflow_adam_optimizer_create(
            ffmodel.handle, alpha, beta1, beta2, weight_decay, epsilon
        )
        self._handle = ffi.gc(self.handle, ffc().flexflow_adam_optimizer_destroy)

    def set_learning_rate(self, learning_rate):
        ffc().flexflow_adam_optimizer_set_lr(self.handle, learning_rate)


# -----------------------------------------------------------------------
# Initializer
# -----------------------------------------------------------------------
class Initializer(object):
    __slots__ = ["handle", "p_handle"]

    def __init__(self, handle, p_handle=0):
        self.p_handle = ffi.new("flexflow_initializer_t *")
        if handle == None:
            self.p_handle.impl = ffi.NULL
        else:
            self.p_handle.impl = handle.impl
        self.handle = self.p_handle[0]
        assert ffi.typeof(self.handle) == ffi.typeof(
            "flexflow_initializer_t"
        ), "Initializer handle is wrong"


# -----------------------------------------------------------------------
# GlorotUniform
# -----------------------------------------------------------------------


class GlorotUniformInitializer(Initializer):
    __slots__ = ["glorot_handle", "_glorot_handle"]

    def __init__(self, seed):
        self.glorot_handle = ffc().flexflow_glorot_uniform_initializer_create(seed)
        self._glorot_handle = ffi.gc(
            self.glorot_handle, ffc().flexflow_glorot_uniform_initializer_destroy
        )
        super(GlorotUniformInitializer, self).__init__(self.glorot_handle)


# -----------------------------------------------------------------------
# ZeroInitializer
# -----------------------------------------------------------------------


class ZeroInitializer(Initializer):
    __slots__ = ["zero_handle", "_zero_handle"]

    def __init__(self):
        self.zero_handle = ffc().flexflow_zero_initializer_create()
        self._zero_handle = ffi.gc(
            self.zero_handle, ffc().flexflow_zero_initializer_destroy
        )
        super(ZeroInitializer, self).__init__(self.zero_handle)


# -----------------------------------------------------------------------
# UniformInitializer
# -----------------------------------------------------------------------


class UniformInitializer(Initializer):
    __slots__ = ["uniform_handle", "_uniform_handle"]

    def __init__(self, seed, minv, maxv):
        self.uniform_handle = ffc().flexflow_uniform_initializer_create(
            seed, minv, maxv
        )
        self._uniform_handle = ffi.gc(
            self.uniform_handle, ffc().flexflow_uniform_initializer_destroy
        )
        super(UniformInitializer, self).__init__(self.uniform_handle)


# -----------------------------------------------------------------------
# NormInitializer
# -----------------------------------------------------------------------


class NormInitializer(Initializer):
    __slots__ = ["norm_handle", "_norm_handle"]

    def __init__(self, seed, mean, stddev):
        self.norm_handle = ffc().flexflow_norm_initializer_create(seed, mean, stddev)
        self._norm_handle = ffi.gc(
            self.norm_handle, ffc().flexflow_norm_initializer_destroy
        )
        super(NormInitializer, self).__init__(self.norm_handle)


# -----------------------------------------------------------------------
# PerfMetrics
# -----------------------------------------------------------------------


class PerfMetrics(object):
    __slots__ = ["handle", "_handle"]

    def __init__(self, handle):
        self.handle = handle
        self._handle = ffi.gc(self.handle, ffc().flexflow_per_metrics_destroy)

    def get_accuracy(self):
        return ffc().flexflow_per_metrics_get_accuracy(self.handle)


# -----------------------------------------------------------------------
# NetConfig
# -----------------------------------------------------------------------


class NetConfig(object):
    def __init__(self):
        self.handle = ffc().flexflow_net_config_create()
        self._handle = ffi.gc(self.handle, ffc().flexflow_net_config_destroy)
        cpath = ffc().flexflow_net_config_get_dataset_path(self.handle)
        self.dataset_path = ffi.string(cpath)


# -----------------------------------------------------------------------
# DLRMConfig
# -----------------------------------------------------------------------


class DLRMConfig(object):
    def __init__(self):
        self.handle = ffc().flexflow_dlrm_config_create()
        self._handle = ffi.gc(self.handle, ffc().flexflow_dlrm_config_destroy)

        cstr = ffc().flexflow_dlrm_config_get_dataset_path(self.handle)
        self.dataset_path = ffi.string(cstr)

        cstr = ffc().flexflow_dlrm_config_get_arch_interaction_op(self.handle)
        self.arch_interaction_op = ffi.string(cstr)

        self.sparse_feature_size = ffc().flexflow_dlrm_config_get_sparse_feature_size(
            self.handle
        )
        self.sigmoid_bot = ffc().flexflow_dlrm_config_get_sigmoid_bot(self.handle)
        self.sigmoid_top = ffc().flexflow_dlrm_config_get_sigmoid_top(self.handle)
        self.embedding_bag_size = ffc().flexflow_dlrm_config_get_embedding_bag_size(
            self.handle
        )
        self.loss_threshold = ffc().flexflow_dlrm_config_get_loss_threshold(self.handle)

        mlp_bot_c = ffc().flexflow_dlrm_config_get_mlp_bot(self.handle)
        self.mlp_bot = []
        for i in range(0, mlp_bot_c[0]):
            self.mlp_bot.append(mlp_bot_c[i + 1])

        mlp_top_c = ffc().flexflow_dlrm_config_get_mlp_top(self.handle)
        self.mlp_top = []
        for i in range(0, mlp_top_c[0]):
            self.mlp_top.append(mlp_top_c[i + 1])

        embedding_size_c = ffc().flexflow_dlrm_config_get_embedding_size(self.handle)
        self.embedding_size = []
        for i in range(0, embedding_size_c[0]):
            self.embedding_size.append(embedding_size_c[i + 1])


# -----------------------------------------------------------------------
# Single DataLoader
# -----------------------------------------------------------------------


class SingleDataLoader(object):
    __slots__ = ["handle", "_handle"]

    def __init__(self, ffmodel, input, full_input, num_samples, data_type):
        assert type(ffmodel) is FFModel, "SingleDataLoader ffmodel is wrong"
        assert type(input) is Tensor, "SingleDataLoader input is wrong"
        if type(full_input) is Tensor:
            self.init_from_tensor(ffmodel, input, full_input, num_samples, data_type)
        else:
            self.init_from_ptr(ffmodel, input, full_input, num_samples, data_type)
        self._handle = ffi.gc(self.handle, ffc().flexflow_single_dataloader_destroy)

    def init_from_tensor(self, ffmodel, input, full_input, num_samples, data_type):
        assert type(full_input) is Tensor, "SingleDataLoader full_input is wrong"
        c_data_type = enum_to_int(DataType, data_type)
        self.handle = ffc().flexflow_single_dataloader_create(
            ffmodel.handle, input.handle, full_input.handle, num_samples, c_data_type
        )

    def init_from_ptr(self, ffmodel, input, full_input, num_samples, data_type):
        # assert type(full_input) is Tensor, "SingleDataLoader full_input is wrong"
        c_data_type = enum_to_int(DataType, data_type)
        self.handle = ffc().flexflow_single_dataloader_create2(
            ffmodel.handle, input.handle, full_input, num_samples, c_data_type
        )

    @property
    def num_samples(self):
        return ffc().flexflow_single_dataloader_get_num_samples(self.handle)

    @num_samples.setter
    def num_samples(self, samples):
        ffc().flexflow_single_dataloader_set_num_samples(self.handle, samples)

    def next_batch(self, ffmodel):
        """Ask the dataloder to load the next batch to the :attr:`batch_tensor`.

        :returns:  None -- no returns.
        """
        ffc().flowflow_single_dataloader_next_batch(self.handle, ffmodel.handle)

    def reset(self):
        """Reset the current position of the dataloder to 0.

        :returns:  None -- no returns.
        """
        ffc().flexflow_single_dataloader_reset(self.handle)


class RegionNdarray(object):
    __slots__ = ["__array_interface__"]

    def __init__(self, shape, data_type, base_ptr, strides, read_only):
        # See: https://docs.scipy.org/doc/numpy/reference/arrays.interface.html
        if data_type == DataType.DT_HALF:
            field_type = "<f2"
        elif data_type == DataType.DT_FLOAT:
            field_type = "<f4"
        elif data_type == DataType.DT_INT32:
            field_type = "<i4"
        else:
            assert 0, "unknown data type"
            field_type = "<f4"
        self.__array_interface__ = {
            "version": 3,
            "shape": shape,
            "typestr": field_type,
            "data": (base_ptr, read_only),
            "strides": strides,
        }


# -----------------------------------------------------------------------
# BatchConfig
# -----------------------------------------------------------------------


class BatchConfig(object):
    __slots__ = ["handle", "_handle"]

    def __init__(self):
        self.handle = ffc().flexflow_batch_config_create()
        self._handle = ffi.gc(self.handle, ffc().flexflow_batch_config_destroy)


# -----------------------------------------------------------------------
# TreeVerifyBatchConfig
# -----------------------------------------------------------------------


class TreeVerifyBatchConfig(object):
    __slots__ = ["handle", "_handle"]

    def __init__(self):
        self.handle = ffc().flexflow_tree_verify_batch_config_create()
        self._handle = ffi.gc(
            self.handle, ffc().flexflow_tree_verify_batch_config_destroy
        )


# -----------------------------------------------------------------------
# BeamSearchBatchConfig
# -----------------------------------------------------------------------


class BatchConfig(object):
    __slots__ = ["handle", "_handle"]

    def __init__(self):
        self.handle = ffc().flexflow_beam_search_batch_config_create()
        self._handle = ffi.gc(
            self.handle, ffc().flexflow_beam_search_batch_config_destroy
        )


# -----------------------------------------------------------------------
# RequestManager
# -----------------------------------------------------------------------


class RequestManager(object):
    __slots__ = ["handle"]

    def __init__(self):
        self.handle = ffc().flexflow_request_manager_get_request_manager()
        # self._handle = ffi.gc(self.handle, ffc().flexflow_request_manager_destroy)

    def register_tokenizer(
        self, model_type, bos_token_id, eos_token_id, tokenizer_filepath
    ):
        c_model_type = enum_to_int(ModelType, model_type)
        c_tokenizer_filepath = get_c_name(tokenizer_filepath)
        return ffc().flexflow_request_manager_register_tokenizer(
            self.handle, c_model_type, bos_token_id, eos_token_id, c_tokenizer_filepath
        )

    def register_output_filepath(self, output_filepath):
        c_output_filepath = get_c_name(output_filepath)
        return ffc().flexflow_request_manager_register_output_filepath(
            self.handle, c_output_filepath
        )

    def register_ssm_model(self, model):
        return ffc().flexflow_request_manager_register_ssm_model(
            self.handle, model.handle
        )

    def set_max_requests_per_batch(self, max_requests):
        return ffc().flexflow_request_manager_set_max_requests_per_batch(
            self.handle, max_requests)
    
    def set_max_tokens_per_batch(self, max_tokens):
        return ffc().flexflow_request_manager_set_max_tokens_per_batch(
            self.handle, max_tokens)
    
    def set_max_sequence_length(self, max_length):
        return ffc().flexflow_request_manager_set_max_sequence_length(
            self.handle, max_length)

    def start_server(self, model):
        return ffc().flexflow_request_manager_start_background_server(
            self.handle, model.handle
        )

    def stop_server(self):
        return ffc().flexflow_request_manager_terminate_background_server(
            self.handle)
# -----------------------------------------------------------------------
# InferenceManager
# -----------------------------------------------------------------------


class InferenceManager(object):
    __slots__ = ["handle"]

    def __init__(self):
        self.handle = ffc().flexflow_inference_manager_get_inference_manager()
        # self._handle = ffi.gc(self.handle, ffc().flexflow_inference_manager_destroy)

    def compile_model_and_allocate_buffer(self, model):
        ffc().flexflow_inference_manager_compile_model_and_allocate_buffer(
            self.handle, model.handle
        )

    def init_operators_inference(self, model):
        ffc().flexflow_inference_manager_init_operators_inference(
            self.handle, model.handle
        )

    def register_model_weights_loader(self, model, fileloader):
        ffc().flexflow_inference_manager_register_model_weights_loader(
            self.handle, model.handle, fileloader.handle
        )

# -----------------------------------------------------------------------
# FileDataLoader
# -----------------------------------------------------------------------


class FileDataLoader(object):
    __slots__ = ["handle", "_handle"]

    def __init__(
        self,
        weight_file_path,
        num_q_heads,
        num_kv_heads,
        hidden_dim,
        qkv_inner_dim,
        tensor_parallelism_degree,
        use_full_precision
    ):
        c_weight_file_path = get_c_name(weight_file_path)
        self.handle = ffc().flexflow_file_data_loader_create(
            c_weight_file_path,
            num_q_heads,
            num_kv_heads,
            hidden_dim,
            qkv_inner_dim,
            tensor_parallelism_degree,
            use_full_precision
        )
        self._handle = ffi.gc(self.handle, ffc().flexflow_file_data_loader_destroy)

    def load_weights(self, model):
        # Check data type and create use_full_precision boolean
        #assert data_type == DataType.DT_FLOAT or data_type == DataType.DT_HALF
        #use_full_precision = data_type == DataType.DT_FLOAT
        ffc().flexflow_file_data_loader_load_weights(
            self.handle, model.handle
        )

# -----------------------------------------------------------------------
# GenerationConfig
# -----------------------------------------------------------------------
        
class GenerationConfig(object):
    """A class to store the sampling configs."""

    def __init__(
        self,
        do_sample: bool = False,
        temperature: float = 0.9,
        topp: float = 0.8,
        topk: int = 1,
    ):
        """Initialize the sampling configs

        :param do_sample: Whether to perform sampling, or use greedy decoding, defaults to False
        :type do_sample: bool, optional
        :param temperature: The temperature setting, defaults to 0.9
        :type temperature: float, optional
        :param topp: The top probabilities (top-p) setting, defaults to 0.8
        :type topp: float, optional
        :param topk: The top-k setting, defaults to 1
        :type topk: int, optional
        """
        self.do_sample = do_sample
        self.temperature = temperature
        self.topp = topp
        self.topk = topk

# -----------------------------------------------------------------------
# GenerationResult
# -----------------------------------------------------------------------

class GenerationResult(object):
    """A class to store the output of a generation request."""

    def __init__(self, text: str = None, tokens: list = None):
        self.output_text = text
        self.output_tokens = tokens

# -----------------------------------------------------------------------
# LoraLinearConfig
# -----------------------------------------------------------------------
        
class LoraLinearConfig(object):
    __slots__ = ["handle", "_handle"]

    def __init__(
        self,
        cache_folder,
        peft_model_id,
    ):
        c_cache_folder = get_c_name(cache_folder)
        peft_model_id = get_c_name(peft_model_id)
        self.handle = ffc().flexflow_lora_linear_config_create(
            c_cache_folder,
            peft_model_id,
        )
        self._handle = ffi.gc(self.handle, ffc().flexflow_lora_linear_config_destroy)

# -----------------------------------------------------------------------
# PEFTModelID
# -----------------------------------------------------------------------
        
class PEFTModelID(object):
    __slots__ = ["handle", "_handle"]

    def __init__(self, id=None):
        if id is None:
            self.handle = ffc().flexflow_peft_model_id_create()
        else:
            self.handle = ffc().flexflow_peft_model_id_create_id(id)
        self._handle = ffi.gc(self.handle, ffc().flexflow_peft_model_id_destroy)

# -----------------------------------------------------------------------
# Request
# -----------------------------------------------------------------------
        
class Request:
    """A class to record the metadata of an inference or finetuning request."""

    def __init__(self, req_type: RequestType, prompt: str = None, max_sequence_length: int = None, peft_model_id: PEFTModelID = None, dataset_filepath: str = None, max_training_steps: int = None):
        self.req_type = req_type
        self.prompt = prompt
        self.max_sequence_length = max_sequence_length
        self.peft_model_id = peft_model_id
        self.dataset_filepath = dataset_filepath
        self.max_training_steps = max_training_steps