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
import subprocess
from enum import Enum

_flexflow_header = subprocess.check_output(['gcc', '-I/home/wwu12/FlexFlow/include', '-E', '-P', '/home/wwu12/FlexFlow/python/flexflow_c.h']).decode('utf-8')
ffi = cffi.FFI()
ffi.cdef(_flexflow_header)
ffc = ffi.dlopen(None)

class ActiMode(Enum):
  AC_MODE_NONE = 10
  AC_MODE_RELU = 11
  AC_MODE_SIGMOID = 12
  AC_MODE_TANH = 13

class PoolType(Enum):
  POOL_MAX = 30
  POOL_AVG = 31
  
class DataType(Enum):
  DT_FLOAT = 40
  DT_DOUBLE = 41
  DT_INT32 = 42
  DT_INT64 = 43
  DT_BOOLEAN = 44

class FFConfig(object):
  def __init__(self):
    self.handle = ffc.flexflow_config_create()
    self._handle = ffi.gc(self.handle, ffc.flexflow_config_destroy)
    
  def parse_args(self):
    ffc.flexflow_config_parse_default_args(self.handle)
    
  def get_batch_size(self):
    return ffc.flexflow_config_get_batch_size(self.handle)
  
  def get_workers_per_node(self):
    return ffc.flexflow_config_get_workers_per_node(self.handle)
  
  def get_num_nodes(self):
    return ffc.flexflow_config_get_num_nodes(self.handle)
    
  def get_epochs(self):
    return ffc.flexflow_config_get_epochs(self.handle)

class Tensor(object):
  def __init__(self, handle):
    self.handle = handle
    self._handle = ffi.gc(self.handle, ffc.flexflow_tensor_4d_destroy)
    
class FFModel(object):
  def __init__(self, config):
    self.handle = ffc.flexflow_model_create(config.handle)
    self._handle = ffi.gc(self.handle, ffc.flexflow_model_destroy)
    
  def create_tensor_4d(self, dims, name, data_type, create_grad=True):
    c_dims = ffi.new("int[]", dims)
    c_data_type = 40
    if (data_type == DataType.DT_FLOAT):
      c_data_type = 40
    elif (data_type == DataType.DT_DOUBLE):
      c_data_type = 41
    elif (data_type == DataType.DT_INT32):
      c_data_type = 42
    elif (data_type == DataType.DT_INT64):
      c_data_type = 43
    elif (data_type == DataType.DT_BOOLEAN):
      c_data_type = 44
    else:
      print("error, unknow data type %d" %(c_data_type))
    handle = ffc.flexflow_tensor_4d_create(self.handle, c_dims, name.encode('utf-8'), c_data_type, create_grad);
    return Tensor(handle)
    
  def conv2d(self, name, input, out_channels, kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w, activation=ActiMode.AC_MODE_NONE):
    c_activation = 10
    if (activation == ActiMode.AC_MODE_NONE):
      c_activation = 10
    elif (activation == ActiMode.AC_MODE_RELU):
      c_activation = 11
    elif (activation == ActiMode.AC_MODE_SIGMOID):
      c_activation = 12
    elif (activation == ActiMode.AC_MODE_TANH):
      c_activation = 13
    else:
      print("error, conv2d unknow activation mode %d" %(activation))
    handle = ffc.flexflow_model_add_conv2d(self.handle, name.encode('utf-8'), input.handle, out_channels, kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w, c_activation)  
    return Tensor(handle)
    
  def pool2d(self, name, input, kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w, pool_type=PoolType.POOL_MAX, relu=True):
    c_pool_type = 30
    if (pool_type == PoolType.POOL_MAX):
      c_pool_type = 30
    elif (pool_type == PoolType.POOL_AVG):
      c_pool_type = 31
    else:
      print("error, pool2d unknow Pool Type %d" %(c_pool_type))
    handle = ffc.flexflow_model_add_pool2d(self.handle, name.encode('utf-8'), input.handle, kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w, c_pool_type, relu)
    return Tensor(handle)
    
  def flat(self, name, input):
    handle = ffc.flexflow_model_add_flat(self.handle, name.encode('utf-8'), input.handle)
    return Tensor(handle)
    
  def linear(self, name, input, out_channels, activation=ActiMode.AC_MODE_NONE, use_bias=True):
    c_activation = 10
    if (activation == ActiMode.AC_MODE_NONE):
      c_activation = 10
    if (activation == ActiMode.AC_MODE_RELU):
      c_activation = 11
    if (activation == ActiMode.AC_MODE_SIGMOID):
      c_activation = 12
    if (activation == ActiMode.AC_MODE_TANH):
      c_activation = 13
    else:
      print("error, linear unknow activation mode %d" %(c_activation))
    handle = ffc.flexflow_model_add_linear(self.handle, name.encode('utf-8'), input.handle, out_channels, c_activation, use_bias)
    return Tensor(handle)
  