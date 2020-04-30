#!/usr/bin/env python

# Copyright 2020 Stanford University
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
from enum import Enum

assert 'FF_DIR' in os.environ
_flexflow_cxxheader_dir= os.path.join(os.environ['FF_DIR'], 'include')
_flexflow_cheader_file = os.path.join(os.path.join(os.environ['FF_DIR'], 'python'), 'flexflow_c.h')

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
  
class OpType(Enum):
  CONV2D = 1011
  EMBEDDING = 1012
  POOL2D = 1013
  LINEAR = 1014
  SOFTMAX = 1015
  CONCAT = 1016
  FLAT = 1017
  
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
  def __init__(self, handle):
    self.handle = handle
  
  def _get_weight_tensor(self):
    handle = ffc.flexflow_op_get_weight(self.handle)
    return Tensor(handle, False)
    
  def _get_bias_tensor(self):
    handle = ffc.flexflow_op_get_bias(self.handle)
    return Tensor(handle, False)
    
  def _get_input_tensor_by_id(self, id):
    handle = ffc.flexflow_op_get_input_by_id(self.handle, id)
    return Tensor(handle, False)
    
  def get_output_tensor(self):
    handle = ffc.flexflow_op_get_output(self.handle)
    return Tensor(handle, False)
    
  def init(self, model):
    ffc.flexflow_op_init(self.handle, model.handle)
    
  def _init_inout(self, model, input):
    handle = ffc.flexflow_op_init_inout(self.handle, model.handle, input.handle)
    return Tensor(handle)
    
  def forward(self, model):
    ffc.flexflow_op_forward(self.handle, model.handle)
    #return Tensor(handle)
    
  def add_to_model(self, model):
    ffc.flexflow_op_add_to_model(self.handle, model.handle)

# -----------------------------------------------------------------------
# Conv2D
# -----------------------------------------------------------------------
class Conv2D(Op):
  def __init__(self, handle):
    super(Conv2D, self).__init__(handle) 
    
  def get_weight_tensor(self):
    return self._get_weight_tensor() 
    
  def get_bias_tensor(self):
    return self._get_bias_tensor() 
    
  def get_input_tensor(self):
    return self._get_input_tensor_by_id(0) 
    
  def init_inout(self, model, input):
    model.add_layer(OpType.CONV2D)
    return self._init_inout(model, input) 
    
# -----------------------------------------------------------------------
# Pool2D
# -----------------------------------------------------------------------
class Pool2D(Op):
  def __init__(self, handle):
    super(Pool2D, self).__init__(handle)
    
  def get_input_tensor(self):
    return self._get_input_tensor_by_id(0) 
    
  def init_inout(self, model, input):
    model.add_layer(OpType.POOL2D)
    return self._init_inout(model, input)

# -----------------------------------------------------------------------
# Linear
# -----------------------------------------------------------------------
class Linear(Op):
  def __init__(self, handle):
    super(Linear, self).__init__(handle)
    
  def get_weight_tensor(self):
    return self._get_weight_tensor() 
    
  def get_bias_tensor(self):
    return self._get_bias_tensor() 
    
  def get_input_tensor(self):
    return self._get_input_tensor_by_id(0) 
    
  def init_inout(self, model, input):
    model.add_layer(OpType.LINEAR)
    return self._init_inout(model, input)

# -----------------------------------------------------------------------
# Flat
# -----------------------------------------------------------------------
class Flat(Op):
  def __init__(self, handle):
    super(Flat, self).__init__(handle)
    
  def get_input_tensor(self):
    return self._get_input_tensor_by_id(0) 
    
  def init_inout(self, model, input):
    model.add_layer(OpType.FLAT)
    return self._init_inout(model, input)
    
# -----------------------------------------------------------------------
# Softmax
# -----------------------------------------------------------------------
class Softmax(Op):
  def __init__(self, handle):
    super(Softmax, self).__init__(handle)
    
# -----------------------------------------------------------------------
# Embedding
# -----------------------------------------------------------------------
class Embedding(Op):
  def __init__(self, handle):
    super(Embedding, self).__init__(handle)
    
# -----------------------------------------------------------------------
# Concat
# -----------------------------------------------------------------------
class Concat(Op):
  def __init__(self, handle):
    super(Concat, self).__init__(handle)
      
# -----------------------------------------------------------------------
# FFConfig
# -----------------------------------------------------------------------

class FFConfig(object):
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
  def __init__(self, handle, deallocate=True):
    self.handle = handle
    self.num_dims = 0
    self.dims = [0, 0, 0, 0]
    self.mapped = False
    self.set_dims()
    if (deallocate == True):
      #print("deallocate true")
      self._handle = ffi.gc(self.handle, ffc.flexflow_tensor_destroy)
    if (self.is_mapped() == True):
      self.mapped = True
      
  def inline_map(self, config):
    assert self.mapped == False, "Tensor is already mapped."
    ffc.flexflow_tensor_inline_map(self.handle, config.handle);
    self.mapped = True
    if (self.num_dims == 0):
      self.set_dims()
    
  def inline_unmap(self, config):
    assert self.mapped == True, "Tensor is not inline mapped."
    ffc.flexflow_tensor_inline_unmap(self.handle, config.handle);
    self.mapped = False
    
  def get_raw_ptr(self, config, data_type):
    if (data_type == DataType.DT_FLOAT):    
      return ffc.flexflow_tensor_get_raw_ptr_float(self.handle, config.handle)
    elif (data_type == DataType.DT_INT32):
      return ffc.flexflow_tensor_get_raw_ptr_int32(self.handle, config.handle)
    else:
      assert 0, "unknown data type"
    
  def get_array(self, config, data_type, layout="O"):
    assert self.mapped == True, "Tensor is not mapped."
    raw_ptr = self.get_raw_ptr(config, data_type)
    raw_ptr_int = int(ffi.cast("uintptr_t", raw_ptr))
    print("raw_ptr: ", raw_ptr, raw_ptr_int)
    strides = None
    if (layout == "O"):
      if (self.num_dims == 1):
        shape = (self.dims[0],)
      elif (self.num_dims == 2):
        shape = (self.dims[0], self.dims[1])
      elif (self.num_dims == 3):
        shape = (self.dims[0], self.dims[1], self.dims[2])
      elif (self.num_dims == 4):
        shape = (self.dims[0], self.dims[1], self.dims[2], self.dims[3])
      else:
        assert 0, "unknow num_dims"
    else:
      datatype_size = get_datatype_size(data_type)
      if (self.num_dims == 1):
        shape = (self.dims[0],)
      elif (self.num_dims == 2):
        shape = (self.dims[1], self.dims[0])
        strides = (datatype_size, self.dims[1]*datatype_size)
      elif (self.num_dims == 3):
        shape = (self.dims[2], self.dims[1], self.dims[0])
        strides = (datatype_size, self.dims[2]*datatype_size, self.dims[2]*self.dims[1]*datatype_size)
      elif (self.num_dims == 4):
        shape = (self.dims[3], self.dims[2], self.dims[1], self.dims[0])
        strides = (datatype_size, self.dims[3]*datatype_size, self.dims[3]*self.dims[2]*datatype_size, self.dims[3]*self.dims[2]*self.dims[1]*datatype_size)
      else:
        assert 0, "unknow num_dims"
    initializer = RegionNdarray(shape, data_type, raw_ptr_int, strides, False)
    array = np.asarray(initializer)
    return array
    
  def set_dims(self):
    self.num_dims = ffc.flexflow_tensor_get_num_dims(self.handle)
    d = ffc.flexflow_tensor_get_dims(self.handle)
    self.dims = [d[0], d[1], d[2], d[3]]
    
  def attach_raw_ptr(self, ffconfig, raw_ptr, column_major=False):
    assert self.mapped == False, "Tensor is already mapped."
    ffc.flexflow_tensor_attach_raw_ptr(self.handle, ffconfig.handle, raw_ptr, column_major)
    self.mapped = True
    
  def detach_raw_ptr(self, ffconfig):
    assert self.mapped == True, "Tensor is not mapped."
    ffc.flexflow_tensor_detach_raw_ptr(self.handle, ffconfig.handle)
    self.mapped = False
    
  def attach_numpy_array(self, ffconfig, np_array):
    np_raw_ptr = np_array.__array_interface__['data']
    print("attach numpy array: ", np_raw_ptr)
    self.attach_raw_ptr(ffconfig, np_raw_ptr[0])
    
  def detach_numpy_array(self, ffconfig):
    self.detach_raw_ptr(ffconfig)
    
  def is_mapped(self):
    return ffc.flexflow_tensor_is_mapped(self.handle)

# -----------------------------------------------------------------------
# FFModel
# -----------------------------------------------------------------------
    
class FFModel(object):
  def __init__(self, config):
    self.handle = ffc.flexflow_model_create(config.handle)
    self._handle = ffi.gc(self.handle, ffc.flexflow_model_destroy)
    self._layers = dict()
    self._nb_layers = 0
    
  def add_layer(self, type):
    self._layers[self._nb_layers] = type
    self._nb_layers += 1
    
  def create_tensor_4d(self, dims, name, data_type, create_grad=True):
    c_dims = ffi.new("int[]", dims)
    c_data_type = enum_to_int(DataType, data_type)
    handle = ffc.flexflow_tensor_4d_create(self.handle, c_dims, name.encode('utf-8'), c_data_type, create_grad);
    return Tensor(handle)
    
  def create_tensor_2d(self, dims, name, data_type, create_grad=True):
    c_dims = ffi.new("int[]", dims)
    c_data_type = enum_to_int(DataType, data_type)
    handle = ffc.flexflow_tensor_2d_create(self.handle, c_dims, name.encode('utf-8'), c_data_type, create_grad);
    return Tensor(handle)
    
  def conv2d(self, name, input, out_channels, kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w, activation=ActiMode.AC_MODE_NONE, use_bias=True):
    c_activation = enum_to_int(ActiMode, activation)
    handle = ffc.flexflow_model_add_conv2d(self.handle, name.encode('utf-8'), input.handle, out_channels, kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w, c_activation, use_bias)  
    self.add_layer(OpType.CONV2D)
    return Tensor(handle)
    
  def conv2d_v2(self, name, in_channels, out_channels, kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w, activation=ActiMode.AC_MODE_NONE, use_bias=True):
    c_activation = enum_to_int(ActiMode, activation)
    handle = ffc.flexflow_model_add_conv2d_no_inout(self.handle, name.encode('utf-8'), in_channels, out_channels, kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w, c_activation, use_bias)  
    return Conv2D(handle)
    
  def embedding(self, name, input, num_entires, out_dim, aggr, kernel_initializer):
    c_aggr = enum_to_int(AggrMode, aggr)
    if type(kernel_initializer) is GlorotUniformInitializer:
      handle = ffc.flexflow_model_add_embedding_with_glorot_uniform_initializer(self.handle, name.encode('utf-8'), input.handle, num_entires, out_dim, c_aggr, kernel_initializer.handle)
    elif type(kernel_initializer) is ZeroInitializer:
      handle = ffc.flexflow_model_add_embedding_with_zero_initializer(self.handle, name.encode('utf-8'), input.handle, num_entires, out_dim, c_aggr, kernel_initializer.handle)
    elif type(kernel_initializer) is UniformInitializer:
      handle = ffc.flexflow_model_add_embedding_with_uniform_initializer(self.handle, name.encode('utf-8'), input.handle, num_entires, out_dim, c_aggr, kernel_initializer.handle)
    elif type(kernel_initializer) is NormInitializer:
      handle = ffc.flexflow_model_add_embedding_with_norm_initializer(self.handle, name.encode('utf-8'), input.handle, num_entires, out_dim, c_aggr, kernel_initializer.handle)
    else:
      assert 0, "unknow initializer type"
    self.add_layer(OpType.EMBEDDING)
    return Tensor(handle)
    
  def pool2d(self, name, input, kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w, pool_type=PoolType.POOL_MAX, activation=ActiMode.AC_MODE_NONE):
    c_pool_type = enum_to_int(PoolType, pool_type)
    c_activation = enum_to_int(ActiMode, activation)
    handle = ffc.flexflow_model_add_pool2d(self.handle, name.encode('utf-8'), input.handle, kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w, c_pool_type, c_activation)
    self.add_layer(OpType.POOL2D)
    return Tensor(handle)
    
  def pool2d_v2(self, name, kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w, pool_type=PoolType.POOL_MAX, activation=ActiMode.AC_MODE_NONE):
    c_pool_type = enum_to_int(PoolType, pool_type)
    c_activation = enum_to_int(ActiMode, activation)
    handle = ffc.flexflow_model_add_pool2d_no_inout(self.handle, name.encode('utf-8'), kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w, c_pool_type, c_activation)
    return Pool2D(handle)

  def dense(self, name, input, out_dim, activation=ActiMode.AC_MODE_NONE, use_bias=True):
    c_activation = enum_to_int(ActiMode, activation)
    handle = ffc.flexflow_model_add_dense_with_default_initializer(self.handle, name.encode('utf-8'), input.handle, out_dim, c_activation, use_bias)
    self.add_layer(OpType.LINEAR)
    return Tensor(handle)
    
  def dense_v2(self, name, in_dim, out_dim, activation=ActiMode.AC_MODE_NONE, use_bias=True):
    c_activation = enum_to_int(ActiMode, activation)
    handle = ffc.flexflow_model_add_dense_with_default_initializer_no_inout(self.handle, name.encode('utf-8'), in_dim, out_dim, c_activation, use_bias)
    return Linear(handle)
    
  def concat(self, name, tensor_list, axis):
    tensor_handle_list = []
    n = 0
    for tensor in tensor_list:
      n = n + 1
      tensor_handle_list.append(tensor.handle)
    handle = ffc.flexflow_model_add_concat(self.handle, name.encode('utf-8'), n, tensor_handle_list, axis)
    self.add_layer(OpType.CONCAT)
    return Tensor(handle)
    
  def flat(self, name, input):
    handle = ffc.flexflow_model_add_flat(self.handle, name.encode('utf-8'), input.handle)
    self.add_layer(OpType.FLAT)
    return Tensor(handle)
    
  def flat_v2(self, name):
    handle = ffc.flexflow_model_add_flat_no_inout(self.handle, name.encode('utf-8'))
    return Flat(handle)
    
  def softmax(self, name, input, label):
    handle = ffc.flexflow_model_add_softmax(self.handle, name.encode('utf-8'), input.handle, label.handle)
    self.add_layer(OpType.SOFTMAX)
    return Tensor(handle)
    
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
    
  def zero_gradients(self):
    ffc.flexflow_model_zero_gradients(self.handle)
  
  def set_sgd_optimizer(self, optimizer):
    ffc.flexflow_model_set_sgd_optimizer(self.handle, optimizer.handle)
  
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
    else:
      assert 0, "unknow layer type"
      return 0
    
  def get_tensor_by_id(self, id):
    handle = ffc.flexflow_model_get_tensor_by_id(self.handle, id)
    return Tensor(handle, False)
    

# -----------------------------------------------------------------------
# SGDOptimizer
# -----------------------------------------------------------------------
    
class SGDOptimizer(object):
  def __init__(self, ffmodel, lr=0.01, momentum=0.0, nesterov=False, weight_decay=0.0):
    self.handle = ffc.flexflow_sgd_optimizer_create(ffmodel.handle, lr, momentum, nesterov, weight_decay)
    self._handle = ffi.gc(self.handle, ffc.flexflow_sgd_optimizer_destroy)  
    
# -----------------------------------------------------------------------
# GlorotUniform
# -----------------------------------------------------------------------

class GlorotUniformInitializer(object):
  def __init__(self, seed):
    self.handle = ffc.flexflow_glorot_uniform_initializer_create(seed)
    self._handle = ffi.gc(self.handle, ffc.flexflow_glorot_uniform_initializer_destroy)
    
# -----------------------------------------------------------------------
# ZeroInitializer
# -----------------------------------------------------------------------

class ZeroInitializer(object):
  def __init__(self):
    self.handle = ffc.flexflow_zero_initializer_create()
    self._handle = ffi.gc(self.handle, ffc.flexflow_zero_initializer_destroy)  
    
# -----------------------------------------------------------------------
# UniformInitializer
# -----------------------------------------------------------------------

class UniformInitializer(object):
  def __init__(self, seed, minv, maxv):
    self.handle = ffc.flexflow_uniform_initializer_create(seed, minv, maxv)
    self._handle = ffi.gc(self.handle, ffc.flexflow_uniform_initializer_destroy)
    
# -----------------------------------------------------------------------
# NormInitializer
# -----------------------------------------------------------------------

class NormInitializer(object):
  def __init__(self, seed, meanv, stddev):
    self.handle = ffc.flexflow_norm_initializer_create(seed, meanv, stddev)
    self._handle = ffi.gc(self.handle, ffc.flexflow_norm_initializer_destroy)

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

class DataLoader(object):
  def __init__(self, ffmodel, ffnetconfig, input, label, full_input=0, full_label=0):
    if (full_input == 0):
      self.handle = ffc.flexflow_dataloader_create(ffmodel.handle, ffnetconfig.handle, input.handle, label.handle)
    else:
      self.handle = ffc.flexflow_dataloader_create_v2(ffmodel.handle, ffnetconfig.handle, input.handle, label.handle, full_input.handle, full_label.handle)
    self._handle = ffi.gc(self.handle, ffc.flexflow_dataloader_destroy)
  
  def set_num_samples(self, samples):
    ffc.flexflow_dataloader_set_num_samples(self.handle, samples)
      
  def get_num_samples(self):
    return ffc.flexflow_dataloader_get_num_samples(self.handle)
    
  def next_batch(self, ffmodel):
    ffc.flowflow_dataloader_next_batch(self.handle, ffmodel.handle)
    
  def reset(self):
    ffc.flexflow_dataloader_reset(self.handle)
    
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

def malloc_int(size):
  return ffc.flexflow_malloc_int(size)
  
def print_array_int(base_ptr, size):
  ffc.flexflow_print_array_int(base_ptr, size)