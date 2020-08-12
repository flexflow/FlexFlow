# Copyright 2020 Stanford University, Los Alamos National Laboratory
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

import cffi
import os
import subprocess
import numpy as np
from .flexflow_logger import fflogger
from enum import Enum

assert 'FF_HOME' in os.environ
_flexflow_cxxheader_dir= os.path.join(os.environ['FF_HOME'], 'include')
_flexflow_cheader_file = os.path.join(os.path.join(os.environ['FF_HOME'], 'python'), 'flexflow_c.h')

_flexflow_cheader = subprocess.check_output(['gcc', '-I', _flexflow_cxxheader_dir, '-E', '-P', _flexflow_cheader_file]).decode('utf-8')
ffi = cffi.FFI()
ffi.cdef(_flexflow_cheader)
ffc = ffi.dlopen(None)

class ActiMode(Enum):
  AC_MODE_NONE = 10
  AC_MODE_RELU = 11
  AC_MODE_SIGMOID = 12
  AC_MODE_TANH = 13
  
class AggrMode(Enum):
  AGGR_MODE_NONE = 20
  AGGR_MODE_SUM = 21
  AGGR_MODE_AVG = 22

class PoolType(Enum):
  POOL_MAX = 30
  POOL_AVG = 31
  
class DataType(Enum):
  DT_FLOAT = 40
  DT_DOUBLE = 41
  DT_INT32 = 42
  DT_INT64 = 43
  DT_BOOLEAN = 44
  
class LossType(Enum):
  LOSS_CATEGORICAL_CROSSENTROPY = 50
  LOSS_SPARSE_CATEGORICAL_CROSSENTROPY = 51
  LOSS_MEAN_SQUARED_ERROR_AVG_REDUCE = 52
  LOSS_MEAN_SQUARED_ERROR_SUM_REDUCE = 53
  
class MetricsType(Enum):
  METRICS_ACCURACY = 1001
  METRICS_CATEGORICAL_CROSSENTROPY = 1002
  METRICS_SPARSE_CATEGORICAL_CROSSENTROPY = 1004
  METRICS_MEAN_SQUARED_ERROR = 1008
  METRICS_ROOT_MEAN_SQUARED_ERROR = 1016
  METRICS_MEAN_ABSOLUTE_ERROR=1032
  
class OpType(Enum):
  CONV2D = 2011
  EMBEDDING = 2012
  POOL2D = 2013
  LINEAR = 2014
  SOFTMAX = 2015
  CONCAT = 2016
  FLAT = 2017
  ELEMENT_UNARY = 2018
  ELEMENT_BINARY = 2019
  MSELOSS = 2020
  BATCH_NORM = 2021
  RELU = 2022
  SIGMOID = 2023
  TANH = 2024
  ELU = 2025
  DROPOUT = 2026
  
def enum_to_int(enum, enum_item):
  for item in enum:
    if (enum_item == item):
      return item.value
  
  assert 0, "unknow enum type " + str(enum_item) + " " + str(enum)    
  return -1
  
def get_datatype_size(datatype):
  if (datatype == DataType.DT_FLOAT):
    return 4
  elif (datatype == DataType.DT_DOUBLE):
    return 8
  elif (datatype == DataType.DT_INT32):
    return 4
  elif (datatype == DataType.DT_INT64):
    return 8
  else:
    assert 0, "unknow datatype" + str(datatype)
    return 0

# -----------------------------------------------------------------------
# Op
# -----------------------------------------------------------------------
class Op(object):
  __slots__ = ['handle']
  def __init__(self, handle):
    assert ffi.typeof(handle) == ffi.typeof('flexflow_op_t'), "Op handle is wrong"
    self.handle = handle

  def _get_parameter_tensor_by_id(self, id):
    handle = ffc.flexflow_op_get_parameter_by_id(self.handle, id)
    return Parameter(handle)
    
  def _get_input_tensor_by_id(self, id):
    handle = ffc.flexflow_op_get_input_by_id(self.handle, id)
    return Tensor(handle, False)

  def _get_output_tensor_by_id(self, id):
    handle = ffc.flexflow_op_get_output_by_id(self.handle, id)
    return Tensor(handle, False)
    
  def init(self, model):
    ffc.flexflow_op_init(self.handle, model.handle)
    
  def _init_inout(self, model, input):
    handle = ffc.flexflow_op_init_inout(self.handle, model.handle, input.handle)
    return Tensor(handle)
    
  def forward(self, model):
    ffc.flexflow_op_forward(self.handle, model.handle)
    #return Tensor(handle)
    
  def _add_to_model(self, model):
    ffc.flexflow_op_add_to_model(self.handle, model.handle)

# -----------------------------------------------------------------------
# ElementBinary
# -----------------------------------------------------------------------
class ElementBinary(Op):
  def __init__(self, handle):
    super(ElementBinary, self).__init__(handle) 
    
# -----------------------------------------------------------------------
# ElementUnary
# -----------------------------------------------------------------------
class ElementUnary(Op):
  def __init__(self, handle):
    super(ElementUnary, self).__init__(handle) 
    
# -----------------------------------------------------------------------
# Conv2D
# -----------------------------------------------------------------------
class Conv2D(Op):
  def __init__(self, handle):
    super(Conv2D, self).__init__(handle) 
    
  def get_weight_tensor(self):
    return self._get_parameter_tensor_by_id(0) 
    
  def get_bias_tensor(self):
    return self._get_parameter_tensor_by_id(1) 
    
  def get_input_tensor(self):
    return self._get_input_tensor_by_id(0) 
    
  def get_output_tensor(self):
    return self._get_output_tensor_by_id(0)
    
  def init_inout(self, model, input):
    model.add_layer(OpType.CONV2D)
    return self._init_inout(model, input) 
    
  def add_to_model(self, model):
    model.add_layer(OpType.CONV2D)
    self._add_to_model(model)
    
# -----------------------------------------------------------------------
# Pool2D
# -----------------------------------------------------------------------
class Pool2D(Op):
  def __init__(self, handle):
    super(Pool2D, self).__init__(handle)
    
  def get_input_tensor(self):
    return self._get_input_tensor_by_id(0) 
    
  def get_output_tensor(self):
    return self._get_output_tensor_by_id(0)
    
  def init_inout(self, model, input):
    model.add_layer(OpType.POOL2D)
    return self._init_inout(model, input)
    
  def add_to_model(self, model):
    model.add_layer(OpType.POOL2D)
    self._add_to_model(model)

# -----------------------------------------------------------------------
# Linear
# -----------------------------------------------------------------------
class Linear(Op):
  def __init__(self, handle):
    super(Linear, self).__init__(handle)
    
  def get_weight_tensor(self):
    return self._get_parameter_tensor_by_id(0) 
    
  def get_bias_tensor(self):
    return self._get_parameter_tensor_by_id(1) 
    
  def get_input_tensor(self):
    return self._get_input_tensor_by_id(0) 
    
  def get_output_tensor(self):
    return self._get_output_tensor_by_id(0)
    
  def init_inout(self, model, input):
    model.add_layer(OpType.LINEAR)
    return self._init_inout(model, input)
    
  def add_to_model(self, model):
    model.add_layer(OpType.LINEAR)
    self._add_to_model(model)

# -----------------------------------------------------------------------
# Flat
# -----------------------------------------------------------------------
class Flat(Op):
  def __init__(self, handle):
    super(Flat, self).__init__(handle)
    
  def get_input_tensor(self):
    return self._get_input_tensor_by_id(0) 
    
  def get_output_tensor(self):
    return self._get_output_tensor_by_id(0)
    
  def init_inout(self, model, input):
    model.add_layer(OpType.FLAT)
    return self._init_inout(model, input)
    
  def add_to_model(self, model):
    model.add_layer(OpType.FLAT)
    self._add_to_model(model)
    
# -----------------------------------------------------------------------
# Softmax
# -----------------------------------------------------------------------
class Softmax(Op):
  def __init__(self, handle):
    super(Softmax, self).__init__(handle)
    
  def add_to_model(self, model):
    model.add_layer(OpType.SOFTMAX)
    self._add_to_model(model)
    
# -----------------------------------------------------------------------
# Embedding
# -----------------------------------------------------------------------
class Embedding(Op):
  def __init__(self, handle):
    super(Embedding, self).__init__(handle)
    
  def add_to_model(self, model):
    model.add_layer(OpType.EMBEDDING)
    self._add_to_model(model)
    
# -----------------------------------------------------------------------
# Concat
# -----------------------------------------------------------------------
class Concat(Op):
  def __init__(self, handle):
    super(Concat, self).__init__(handle)
    
  def add_to_model(self, model):
    model.add_layer(OpType.CONCAT)
    self._add_to_model(model)
    
# -----------------------------------------------------------------------
# MSELoss
# -----------------------------------------------------------------------
class MSELoss(Op):
  def __init__(self, handle):
    super(MSELoss, self).__init__(handle)
    
# -----------------------------------------------------------------------
# BatchNorm
# -----------------------------------------------------------------------
class BatchNorm(Op):
  def __init__(self, handle):
    super(BatchNorm, self).__init__(handle)
    
# -----------------------------------------------------------------------
# Dropout
# -----------------------------------------------------------------------
class Dropout(Op):
  def __init__(self, handle):
    super(Dropout, self).__init__(handle)
    
# -----------------------------------------------------------------------
# Relu
# -----------------------------------------------------------------------
class Relu(Op):
  def __init__(self, handle):
    super(Relu, self).__init__(handle)
    
# -----------------------------------------------------------------------
# Sigmod
# -----------------------------------------------------------------------
class Sigmoid(Op):
  def __init__(self, handle):
    super(Sigmoid, self).__init__(handle)
    
# -----------------------------------------------------------------------
# Relu
# -----------------------------------------------------------------------
class Tanh(Op):
  def __init__(self, handle):
    super(Tanh, self).__init__(handle)
    
# -----------------------------------------------------------------------
# Elu
# -----------------------------------------------------------------------
class Elu(Op):
  def __init__(self, handle):
    super(Elu, self).__init__(handle)
      
# -----------------------------------------------------------------------
# FFConfig
# -----------------------------------------------------------------------

class FFConfig(object):
  __slots__ = ['handle', '_handle']
  def __init__(self):
    self.handle = ffc.flexflow_config_create()
    self._handle = ffi.gc(self.handle, ffc.flexflow_config_destroy)
    
  def parse_args(self):
    ffc.flexflow_config_parse_args_default(self.handle)
    
  def get_batch_size(self):
    return ffc.flexflow_config_get_batch_size(self.handle)
  
  def get_workers_per_node(self):
    return ffc.flexflow_config_get_workers_per_node(self.handle)
  
  def get_num_nodes(self):
    return ffc.flexflow_config_get_num_nodes(self.handle)
    
  def get_epochs(self):
    return ffc.flexflow_config_get_epochs(self.handle)
  
  def get_current_time(self):
    return ffc.flexflow_get_current_time(self.handle)
    
  def begin_trace(self, trace_id):
    ffc.flexflow_begin_trace(self.handle, trace_id)
    
  def end_trace(self, trace_id):
    ffc.flexflow_end_trace(self.handle, trace_id)

# -----------------------------------------------------------------------
# Tensor
# -----------------------------------------------------------------------

class Tensor(object):
  __slots__ = ['p_handle', 'handle', '_handle', 'num_dims', 'dims', 'data_type', 'mapped']
  def __init__(self, handle, deallocate=True):
    if (ffi.typeof(handle) == ffi.typeof('flexflow_tensor_t')):
      self.p_handle = 0
      self.handle = handle
    elif (ffi.typeof(handle) == ffi.typeof('flexflow_parameter_t')):
      self.p_handle = ffi.new('flexflow_tensor_t *')
      self.p_handle.impl = handle.impl
      self.handle = self.p_handle[0]
    else:
      assert 0, "Tensor handle is wrong"
    self.num_dims = 0
    self.dims = 0
    self.mapped = False
    self.__get_dims()
    self.__get_data_type()
    if (deallocate == True):
      self._handle = ffi.gc(self.handle, ffc.flexflow_tensor_destroy)
    if (self.is_mapped() == True):
      self.mapped = True
      
  def inline_map(self, ffconfig):
    assert self.mapped == False, "Tensor is already mapped."
    ffc.flexflow_tensor_inline_map(self.handle, ffconfig.handle);
    self.mapped = True
    assert self.num_dims > 0, "check dims"
    
  def inline_unmap(self, ffconfig):
    assert self.mapped == True, "Tensor is not inline mapped."
    ffc.flexflow_tensor_inline_unmap(self.handle, ffconfig.handle);
    self.mapped = False
    
  def get_array(self, ffconfig, data_type):
    assert self.mapped == True, "Tensor is not mapped."
    raw_ptr = self.__get_raw_ptr(ffconfig, data_type)
    raw_ptr_int = int(ffi.cast("uintptr_t", raw_ptr))
    fflogger.debug("raw_ptr: %s, %d" %( str(raw_ptr), raw_ptr_int))
    strides = None
    if (self.num_dims >= 1 or self.num_dims <= 4):
      shape = self.dims
    else:
      assert 0, "unknow num_dims"
    initializer = RegionNdarray(shape, data_type, raw_ptr_int, strides, False)
    array = np.asarray(initializer)
    return array
  
  def get_flat_array(self, ffconfig, data_type):
    assert self.mapped == True, "Tensor is not mapped."
    raw_ptr = self.__get_raw_ptr(ffconfig, data_type)
    raw_ptr_int = int(ffi.cast("uintptr_t", raw_ptr))
    fflogger.debug("raw_ptr: %s, %d" %( str(raw_ptr), raw_ptr_int))
    strides = None
    if (self.num_dims >= 1 or self.num_dims <= 4):
      shape_prod = np.prod(self.dims)
      shape = (shape_prod,)
    else:
      assert 0, "unknown num_dims"
    initializer = RegionNdarray(shape, data_type, raw_ptr_int, strides, False)
    array = np.asarray(initializer)
    return array
    
  def attach_numpy_array(self, ffconfig, np_array):
    assert np_array.__array_interface__['strides'] == None, "numpy array strides is not None"
    np_shape = np_array.shape
    num_dims = len(np_shape)
    assert num_dims == self.num_dims, "please check dims (%d == %d)" %(num_dims, self.num_dims)
    for i in range(0, num_dims):
      assert np_shape[i] == self.dims[i], "please check shape dim %d (%d == %d)" %(i, np_shape[i], self.dims[i])
    np_raw_ptr = np_array.__array_interface__['data']
    raw_ptr = ffi.cast("void*", np_raw_ptr[0])
    fflogger.debug("attach numpy array: %s, %s, %s" %( str(np_raw_ptr), str(raw_ptr), hex(np_raw_ptr[0])))
    self.__attach_raw_ptr(ffconfig, raw_ptr)
    
  def detach_numpy_array(self, ffconfig):
    self.__detach_raw_ptr(ffconfig)
    
  def is_mapped(self):
    return ffc.flexflow_tensor_is_mapped(self.handle)
    
  def __get_raw_ptr(self, ffconfig, data_type):
    assert data_type == self.data_type, "Tensor check data type"
    if (data_type == DataType.DT_FLOAT):    
      return ffc.flexflow_tensor_get_raw_ptr_float(self.handle, ffconfig.handle)
    elif (data_type == DataType.DT_INT32):
      return ffc.flexflow_tensor_get_raw_ptr_int32(self.handle, ffconfig.handle)
    else:
      assert 0, "unknown data type"
    
  def __get_dims(self):
    self.num_dims = ffc.flexflow_tensor_get_num_dims(self.handle)
    d = ffc.flexflow_tensor_get_dims(self.handle)
    #fflogger.debug(d[0], d[1], d[2], d[3])
    if (self.num_dims == 1):
      self.dims = (d[0],)
    elif (self.num_dims == 2):
      self.dims = (d[1], d[0])
    elif (self.num_dims == 3):
      self.dims = (d[2], d[1], d[0])
    elif (self.num_dims == 4):
      self.dims = (d[3], d[2], d[1], d[0])
    else:
      assert 0, "unknown num_dims"
      
  def __get_data_type(self):
    dtype = ffc.flexflow_tensor_get_data_type(self.handle)
    if (dtype == 40):
      self.data_type = DataType.DT_FLOAT
    elif (dtype == 41):
      self.data_type = DataType.DT_DOUBLE
    elif (dtype == 42):
      self.data_type = DataType.DT_INT32
    elif (dtype == 43):
      self.data_type = DataType.DT_INT64
    elif (dtype == 44):
      self.data_type = DataType.DT_BOOLEAN
    else:
      assert 0, "unknown data type"
    
  def __attach_raw_ptr(self, ffconfig, raw_ptr, column_major=True):
    assert self.mapped == False, "Tensor is already mapped."
    ffc.flexflow_tensor_attach_raw_ptr(self.handle, ffconfig.handle, raw_ptr, column_major)
    self.mapped = True
    
  def __detach_raw_ptr(self, ffconfig):
    assert self.mapped == True, "Tensor is not mapped."
    ffc.flexflow_tensor_detach_raw_ptr(self.handle, ffconfig.handle)
    self.mapped = False
    
# -----------------------------------------------------------------------
# Parameter
# -----------------------------------------------------------------------

class Parameter(Tensor):
  __slots__ = ['parameter_handle']
  def __init__(self, handle):
    assert ffi.typeof(handle) == ffi.typeof('flexflow_parameter_t'), "Parameter handle is wrong"
    self.parameter_handle = handle
    super(Parameter, self).__init__(self.parameter_handle, deallocate=False)
    
  def set_weights(self, ffmodel, np_array):
    assert np_array.__array_interface__['strides'] == None, "Parameter set_weights, numpy array strides is not None"
    np_shape = np_array.shape
    num_dims = len(np_shape)
    assert num_dims == self.num_dims, "please check dims (%d == %d)" %(num_dims, self.num_dims)
    for i in range(0, num_dims):
      assert np_shape[i] == self.dims[i], "please check shape dim %d (%d == %d)" %(i, np_shape[i], self.dims[i])
    c_dims = ffi.new("int[]", self.dims)
    np_raw_ptr = np_array.__array_interface__['data']
    raw_ptr = ffi.cast("float*", np_raw_ptr[0])
    fflogger.debug("set weights raw_ptr: %s, %s, %s, %s" %( str(raw_ptr), str(np_raw_ptr[0]), hex(np_raw_ptr[0]), str(np_shape)))
    ret_val = ffc.flexflow_parameter_set_weights_float(self.parameter_handle, ffmodel.handle, num_dims, c_dims, raw_ptr)
    assert ret_val == True, ret_val
    
  def get_weights(self, ffmodel):
    shape = self.dims
    np_array = np.empty(shape, dtype=np.float32)
    np_raw_ptr = np_array.__array_interface__['data']
    raw_ptr = ffi.cast("float*", np_raw_ptr[0])
    fflogger.debug("get weights raw_ptr: %s, %s, %s, %s" %( str(raw_ptr), str(np_raw_ptr[0]), hex(np_raw_ptr[0]), str(shape)))
    ret_val = ffc.flexflow_parameter_get_weights_float(self.parameter_handle, ffmodel.handle, raw_ptr)
    assert ret_val == True
    return np_array
    
# -----------------------------------------------------------------------
# FFModel
# -----------------------------------------------------------------------
    
class FFModel(object):
  __slots__ = ['handle', '_handle', '_layers', '_nb_layers']
  def __init__(self, ffconfig):
    self.handle = ffc.flexflow_model_create(ffconfig.handle)
    self._handle = ffi.gc(self.handle, ffc.flexflow_model_destroy)
    self._layers = dict()
    self._nb_layers = 0
    
  def get_layers(self):
    return self._layers
    
  def add_layer(self, type):
    self._layers[self._nb_layers] = type
    self._nb_layers += 1

  def create_tensor(self, dims, name, data_type, create_grad=True):
    c_dims = ffi.new("int[]", dims)
    c_data_type = enum_to_int(DataType, data_type)
    num_dims = len(dims)
    handle = ffc.flexflow_tensor_create(self.handle, num_dims, c_dims, name.encode('utf-8'), c_data_type, create_grad);
    return Tensor(handle)
    
  def exp(self, x):
    handle = ffc.flexflow_model_add_exp(self.handle, x.handle)
    self.add_layer(OpType.ELEMENT_UNARY)
    return Tensor(handle)
    
  def add(self, x, y):
    handle = ffc.flexflow_model_add_add(self.handle, x.handle, y.handle)
    self.add_layer(OpType.ELEMENT_BINARY)
    return Tensor(handle)
  
  def subtract(self, x, y):
    handle = ffc.flexflow_model_add_subtract(self.handle, x.handle, y.handle)
    self.add_layer(OpType.ELEMENT_BINARY)
    return Tensor(handle)
    
  def multiply(self, x, y):
    handle = ffc.flexflow_model_add_multiply(self.handle, x.handle, y.handle)
    self.add_layer(OpType.ELEMENT_BINARY)
    return Tensor(handle)
    
  def divide(self, x, y):
    handle = ffc.flexflow_model_add_divide(self.handle, x.handle, y.handle)
    self.add_layer(OpType.ELEMENT_BINARY)
    return Tensor(handle)
    
  def conv2d(self, input, out_channels, kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w, activation=ActiMode.AC_MODE_NONE, use_bias=True, kernel_initializer=None, bias_initializer=None):
    c_activation = enum_to_int(ActiMode, activation)
    kernel_init_handle = self.__get_initializer_handle(kernel_initializer)
    bias_init_handle = self.__get_initializer_handle(bias_initializer)
    handle = ffc.flexflow_model_add_conv2d(self.handle, input.handle, out_channels, kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w, c_activation, use_bias, kernel_init_handle, bias_init_handle)  
    self.add_layer(OpType.CONV2D)
    return Tensor(handle)
    
  def conv2d_v2(self, in_channels, out_channels, kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w, activation=ActiMode.AC_MODE_NONE, use_bias=True, kernel_initializer=None, bias_initializer=None):
    c_activation = enum_to_int(ActiMode, activation)
    kernel_init_handle = self.__get_initializer_handle(kernel_initializer)
    bias_init_handle = self.__get_initializer_handle(bias_initializer)
    handle = ffc.flexflow_model_add_conv2d_no_inout(self.handle, in_channels, out_channels, kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w, c_activation, use_bias, kernel_init_handle, bias_init_handle)  
    return Conv2D(handle)
    
  def embedding(self, input, num_entires, out_dim, aggr, kernel_initializer):
    c_aggr = enum_to_int(AggrMode, aggr)
    assert (type(kernel_initializer) is GlorotUniformInitializer) or (type(kernel_initializer) is ZeroInitializer) or (type(kernel_initializer) is UniformInitializer) or (type(kernel_initializer) is NormInitializer), "unknow initializer type"
    handle = ffc.flexflow_model_add_embedding(self.handle,  input.handle, num_entires, out_dim, c_aggr, kernel_initializer.handle)
    self.add_layer(OpType.EMBEDDING)
    return Tensor(handle)
    
  def pool2d(self, input, kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w, pool_type=PoolType.POOL_MAX, activation=ActiMode.AC_MODE_NONE):
    c_pool_type = enum_to_int(PoolType, pool_type)
    c_activation = enum_to_int(ActiMode, activation)
    handle = ffc.flexflow_model_add_pool2d(self.handle, input.handle, kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w, c_pool_type, c_activation)
    self.add_layer(OpType.POOL2D)
    return Tensor(handle)
    
  def pool2d_v2(self, kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w, pool_type=PoolType.POOL_MAX, activation=ActiMode.AC_MODE_NONE):
    c_pool_type = enum_to_int(PoolType, pool_type)
    c_activation = enum_to_int(ActiMode, activation)
    handle = ffc.flexflow_model_add_pool2d_no_inout(self.handle, kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w, c_pool_type, c_activation)
    return Pool2D(handle)
    
  def batch_norm(self, input, relu=True):
    handle = ffc.flexflow_model_add_batch_norm(self.handle, name.encode('utf-8'), input.handle, relu)
    self.add_layer(OpType.BATCH_NORM)
    return Tensor(handle)

  def dense(self, input, out_dim, activation=ActiMode.AC_MODE_NONE, use_bias=True, kernel_initializer=None, bias_initializer=None):
    c_activation = enum_to_int(ActiMode, activation)
    kernel_init_handle = self.__get_initializer_handle(kernel_initializer)
    bias_init_handle = self.__get_initializer_handle(bias_initializer)
    handle = ffc.flexflow_model_add_dense(self.handle,  input.handle, out_dim, c_activation, use_bias, kernel_init_handle, bias_init_handle)
    self.add_layer(OpType.LINEAR)
    return Tensor(handle)
    
  def dense_v2(self, in_dim, out_dim, activation=ActiMode.AC_MODE_NONE, use_bias=True, kernel_initializer=None, bias_initializer=None):
    c_activation = enum_to_int(ActiMode, activation)
    kernel_init_handle = self.__get_initializer_handle(kernel_initializer)
    bias_init_handle = self.__get_initializer_handle(bias_initializer)
    handle = ffc.flexflow_model_add_dense_no_inout(self.handle,  in_dim, out_dim, c_activation, use_bias, kernel_init_handle, bias_init_handle)
    return Linear(handle)
    
  def concat(self, tensor_list, axis):
    tensor_handle_list = []
    n = 0
    for tensor in tensor_list:
      n = n + 1
      tensor_handle_list.append(tensor.handle)
    c_tensor_handle_list = ffi.new("flexflow_tensor_t[]", tensor_handle_list)
    handle = ffc.flexflow_model_add_concat(self.handle, n, c_tensor_handle_list, axis)
    self.add_layer(OpType.CONCAT)
    return Tensor(handle)
    
  def flat(self, input):
    handle = ffc.flexflow_model_add_flat(self.handle, input.handle)
    self.add_layer(OpType.FLAT)
    return Tensor(handle)
    
  def flat_v2(self):
    handle = ffc.flexflow_model_add_flat_no_inout(self.handle)
    return Flat(handle)
    
  def softmax(self, input):
    handle = ffc.flexflow_model_add_softmax(self.handle, input.handle)
    self.add_layer(OpType.SOFTMAX)
    return Tensor(handle)
    
  def relu(self, input):
    handle = ffc.flexflow_model_add_relu(self.handle, input.handle)
    self.add_layer(OpType.RELU)
    return Tensor(handle)
    
  def sigmoid(self, input):
    handle = ffc.flexflow_model_add_sigmoid(self.handle, input.handle)
    self.add_layer(OpType.SIGMOID)
    return Tensor(handle)
    
  def tanh(self, input):
    handle = ffc.flexflow_model_add_tanh(self.handle, input.handle)
    self.add_layer(OpType.TANH)
    return Tensor(handle)
    
  def elu(self, input):
    handle = ffc.flexflow_model_add_elu(self.handle, input.handle)
    self.add_layer(OpType.ELU)
    return Tensor(handle)
    
  def dropout(self, input, rate, seed):
    handle = ffc.flexflow_model_add_dropout(self.handle, input.handle, rate, seed)
    self.add_layer(OpType.DROPOUT)
    return Tensor(handle)
    
  # def mse_loss(self, logits, labels, reduction):
  #   ffc.flexflow_model_add_mse_loss(self.handle, logits.handle, labels.handle, reduction.encode('utf-8'))
  #   self.add_layer(OpType.MSELOSS)
    
  def reset_metrics(self):
    ffc.flexflow_model_reset_metrics(self.handle)
    
  def init_layers(self):
    ffc.flexflow_model_init_layers(self.handle)
    
  def prefetch(self):
    ffc.flexflow_model_prefetch(self.handle)
    
  def forward(self):
    ffc.flexflow_model_forward(self.handle)
    
  def backward(self):
    ffc.flexflow_model_backward(self.handle)
    
  def update(self):
    ffc.flexflow_model_update(self.handle)
    
  def compile(self, optimizer=None, loss_type=None, metrics=None):
    if isinstance(optimizer, SGDOptimizer) == True:
      self.set_sgd_optimizer(optimizer)
    elif isinstance(optimizer, AdamOptimizer) == True:
      self.set_adam_optimizer(optimizer)
    elif optimizer == None:
      pass
    else:
      assert 0, "[Model]: unknown optimizer"
      
    c_loss_type = enum_to_int(LossType, loss_type)
    metrics_int = []
    for metric in metrics:
      metrics_int.append(enum_to_int(MetricsType, metric))
    c_metrics = ffi.new("int[]", metrics_int)
    ffc.flexflow_model_compile(self.handle, c_loss_type, c_metrics, len(metrics))
    
  def zero_gradients(self):
    ffc.flexflow_model_zero_gradients(self.handle)
  
  def set_sgd_optimizer(self, optimizer):
    ffc.flexflow_model_set_sgd_optimizer(self.handle, optimizer.handle)
    
  def set_adam_optimizer(self, optimizer):
    ffc.flexflow_model_set_adam_optimizer(self.handle, optimizer.handle)
  
  def print_layers(self, id=-1):
    ffc.flexflow_model_print_layers(self.handle, id)
    
  def get_layer_by_id(self, layer_id):
    handle = ffc.flexflow_model_get_layer_by_id(self.handle, layer_id)
    if (self._layers[layer_id] == OpType.CONV2D):
      return Conv2D(handle)
    elif (self._layers[layer_id] == OpType.POOL2D):
      return Pool2D(handle)
    elif (self._layers[layer_id] == OpType.LINEAR):
      return Linear(handle)
    elif (self._layers[layer_id] == OpType.EMBEDDING):
      return Embedding(handle)
    elif (self._layers[layer_id] == OpType.FLAT):
      return Flat(handle)
    elif (self._layers[layer_id] == OpType.CONCAT):
      return Concat(handle)
    elif (self._layers[layer_id] == OpType.SOFTMAX):
      return Softmax(handle)
    elif (self._layers[layer_id] == OpType.ELEMENT_UNARY):
      return ElementUnary(handle)
    elif (self._layers[layer_id] == OpType.ELEMENT_BINARY):
      return ElementBinary(handle)
    elif (self._layers[layer_id] == OpType.MSELOSS):
      return MSELoss(handle)
    elif (self._layers[layer_id] == OpType.RELU):
      return Dropout(handle)
    elif (self._layers[layer_id] == OpType.SIGMOID):
      return Dropout(handle)
    elif (self._layers[layer_id] == OpType.TANH):
      return Dropout(handle)
    elif (self._layers[layer_id] == OpType.ELU):
      return Dropout(handle)
    elif (self._layers[layer_id] == OpType.DROPOUT):
      return Dropout(handle)
    else:
      assert 0, "unknow layer type"
      return 0
    
  def get_tensor_by_id(self, id):
    handle = ffc.flexflow_model_get_parameter_by_id(self.handle, id)
    return Parameter(handle)
    
  def get_label_tensor(self):
    handle = ffc.flexflow_model_get_label_tensor(self.handle)
    return Tensor(handle, deallocate=False)
    
  def get_perf_metrics(self):
    handle = ffc.flexflow_model_get_perf_metrics(self.handle)
    return PerfMetrics(handle)
    
  def __get_initializer_handle(self, initializer):
    if (initializer == None):
      null_initializer = Initializer(None)
      return null_initializer.handle
    else:
      return initializer.handle

# -----------------------------------------------------------------------
# SGDOptimizer
# -----------------------------------------------------------------------
    
class SGDOptimizer(object):
  __slots__ = ['handle', '_handle']
  def __init__(self, ffmodel, lr=0.01, momentum=0.0, nesterov=False, weight_decay=0.0):
    self.handle = ffc.flexflow_sgd_optimizer_create(ffmodel.handle, lr, momentum, nesterov, weight_decay)
    self._handle = ffi.gc(self.handle, ffc.flexflow_sgd_optimizer_destroy)  
  
  def set_learning_rate(self, learning_rate):
    ffc.flexflow_sgd_optimizer_set_lr(self.handle, learning_rate)
    
# -----------------------------------------------------------------------
# AdamOptimizer
# -----------------------------------------------------------------------
    
class AdamOptimizer(object):
  __slots__ = ['handle', '_handle']
  def __init__(self, ffmodel, alpha=0.001, beta1=0.9, beta2=0.999, weight_decay=0.0, epsilon=1e-8):
    self.handle = ffc.flexflow_adam_optimizer_create(ffmodel.handle, alpha, beta1, beta2, weight_decay, epsilon)
    self._handle = ffi.gc(self.handle, ffc.flexflow_adam_optimizer_destroy)
    
  def set_learning_rate(self, learning_rate):
    ffc.flexflow_adam_optimizer_set_lr(self.handle, learning_rate)  

# -----------------------------------------------------------------------
# Initializer
# -----------------------------------------------------------------------
class Initializer(object):
  __slots__ = ['handle', 'p_handle']
  def __init__(self, handle, p_handle=0):
    self.p_handle = ffi.new('flexflow_initializer_t *')
    if (handle == None):
      self.p_handle.impl = ffi.NULL
    else:
      self.p_handle.impl = handle.impl
    self.handle = self.p_handle[0]
    assert ffi.typeof(self.handle) == ffi.typeof('flexflow_initializer_t'), "Initializer handle is wrong"
      
# -----------------------------------------------------------------------
# GlorotUniform
# -----------------------------------------------------------------------

class GlorotUniformInitializer(Initializer):
  __slots__ = ['glorot_handle', '_glorot_handle']
  def __init__(self, seed):
    self.glorot_handle = ffc.flexflow_glorot_uniform_initializer_create(seed)
    self._glorot_handle = ffi.gc(self.glorot_handle, ffc.flexflow_glorot_uniform_initializer_destroy)
    super(GlorotUniformInitializer, self).__init__(self.glorot_handle)
    
# -----------------------------------------------------------------------
# ZeroInitializer
# -----------------------------------------------------------------------

class ZeroInitializer(Initializer):
  __slots__ = ['zero_handle', '_zero_handle']
  def __init__(self):
    self.zero_handle = ffc.flexflow_zero_initializer_create()
    self._zero_handle = ffi.gc(self.zero_handle, ffc.flexflow_zero_initializer_destroy)
    super(ZeroInitializer, self).__init__(self.zero_handle)  
    
# -----------------------------------------------------------------------
# UniformInitializer
# -----------------------------------------------------------------------

class UniformInitializer(Initializer):
  __slots__ = ['uniform_handle', '_uniform_handle']
  def __init__(self, seed, minv, maxv):
    self.uniform_handle = ffc.flexflow_uniform_initializer_create(seed, minv, maxv)
    self._uniform_handle = ffi.gc(self.uniform_handle, ffc.flexflow_uniform_initializer_destroy)
    super(UniformInitializer, self).__init__(self.uniform_handle)  
    
# -----------------------------------------------------------------------
# NormInitializer
# -----------------------------------------------------------------------

class NormInitializer(Initializer):
  __slots__ = ['norm_handle', '_norm_handle']
  def __init__(self, seed, meanv, stddev):
    self.norm_handle = ffc.flexflow_norm_initializer_create(seed, meanv, stddev)
    self._norm_handle = ffi.gc(self.norm_handle, ffc.flexflow_norm_initializer_destroy)
    super(NormInitializer, self).__init__(self.norm_handle)  

# -----------------------------------------------------------------------
# PerfMetrics
# -----------------------------------------------------------------------

class PerfMetrics(object):
  __slots__= ['handle', '_handle']
  def __init__(self, handle):
    self.handle = handle
    self._handle = ffi.gc(self.handle, ffc.flexflow_per_metrics_destroy)
    
  def get_accuracy(self):
    return ffc.flexflow_per_metrics_get_accuracy(self.handle)

# -----------------------------------------------------------------------
# NetConfig
# -----------------------------------------------------------------------

class NetConfig(object):
  def __init__(self):
    self.handle = ffc.flexflow_net_config_create()
    self._handle = ffi.gc(self.handle, ffc.flexflow_net_config_destroy)
    cpath = ffc.flexflow_net_config_get_dataset_path(self.handle)
    self.dataset_path = ffi.string(cpath)
    
# -----------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------

class DataLoader4D(object):
  __slots__ = ['handle', '_handle']
  def __init__(self, ffmodel, input, label, full_input=0, full_label=0, num_samples=0, ffnetconfig=0):
    if (ffnetconfig == 0):
      self.handle = ffc.flexflow_dataloader_4d_create_v2(ffmodel.handle, input.handle, label.handle, full_input.handle, full_label.handle, num_samples)
    else:
      self.handle = ffc.flexflow_dataloader_4d_create(ffmodel.handle, ffnetconfig.handle, input.handle, label.handle)
    self._handle = ffi.gc(self.handle, ffc.flexflow_dataloader_4d_destroy)
  
  def set_num_samples(self, samples):
    ffc.flexflow_dataloader_4d_set_num_samples(self.handle, samples)
      
  def get_num_samples(self):
    return ffc.flexflow_dataloader_4d_get_num_samples(self.handle)
    
  def next_batch(self, ffmodel):
    ffc.flowflow_dataloader_4d_next_batch(self.handle, ffmodel.handle)
    
  def reset(self):
    ffc.flexflow_dataloader_4d_reset(self.handle)
    
class DataLoader2D(object):
  __slots__ = ['handle', '_handle']
  def __init__(self, ffmodel, input, label, full_input=0, full_label=0, num_samples=0):
    self.handle = ffc.flexflow_dataloader_2d_create_v2(ffmodel.handle, input.handle, label.handle, full_input.handle, full_label.handle, num_samples)
    self._handle = ffi.gc(self.handle, ffc.flexflow_dataloader_2d_destroy)
  
  def set_num_samples(self, samples):
    ffc.flexflow_dataloader_2d_set_num_samples(self.handle, samples)
      
  def get_num_samples(self):
    return ffc.flexflow_dataloader_2d_get_num_samples(self.handle)
    
  def next_batch(self, ffmodel):
    ffc.flowflow_dataloader_2d_next_batch(self.handle, ffmodel.handle)
    
  def reset(self):
    ffc.flexflow_dataloader_2d_reset(self.handle)
    
# -----------------------------------------------------------------------
# Single DataLoader
# -----------------------------------------------------------------------

class SingleDataLoader(object):
  __slots__ = ['handle', '_handle']
  def __init__(self, ffmodel, input, full_input, num_samples, data_type):
    assert type(ffmodel) is FFModel, "SingleDataLoader ffmodel is wrong"
    assert type(input) is Tensor, "SingleDataLoader input is wrong"
    assert type(full_input) is Tensor, "SingleDataLoader full_input is wrong"
    c_data_type = enum_to_int(DataType, data_type)
    self.handle = ffc.flexflow_single_dataloader_create(ffmodel.handle, input.handle, full_input.handle, num_samples, c_data_type)
    self._handle = ffi.gc(self.handle, ffc.flexflow_single_dataloader_destroy)
  
  def set_num_samples(self, samples):
    ffc.flexflow_single_dataloader_set_num_samples(self.handle, samples)
      
  def get_num_samples(self):
    return ffc.flexflow_single_dataloader_get_num_samples(self.handle)
    
  def next_batch(self, ffmodel):
    ffc.flowflow_single_dataloader_next_batch(self.handle, ffmodel.handle)
    
  def reset(self):
    ffc.flexflow_single_dataloader_reset(self.handle)
    
class RegionNdarray(object):
  __slots__ = ['__array_interface__']
  def __init__(self, shape, data_type, base_ptr, strides, read_only):
    # See: https://docs.scipy.org/doc/numpy/reference/arrays.interface.html
    if (data_type == DataType.DT_FLOAT):    
      field_type = "<f4"
    elif (data_type == DataType.DT_INT32):
      field_type = "<i4"
    else:
      assert 0, "unknown data type"
      field_type = "<f4"
    self.__array_interface__ = {
      'version': 3,
      'shape': shape,
      'typestr': field_type,
      'data': (base_ptr, read_only),
      'strides': strides,
    }
