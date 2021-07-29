# Copyright 2021 Stanford University, Los Alamos National Laboratory
#                Facebook
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

import numpy as np
from .flexflow_logger import fflogger
# from .flexflow_type import ActiMode, AggrMode, PoolType, DataType, LossType, CompMode, MetricsType, OpType, ParameterSyncType, enum_to_int, int_to_enum
from .flexflow_pybind11_internal import ActiMode, CompMode, DataType, LossType, MetricsType, PoolType, ParameterSyncType
from .flexflow_pybind11_internal import Initializer, GlorotUniformInitializer, UniformInitializer, ZeroInitializer
from .flexflow_pybind11_internal import Optimizer, SGDOptimizer, AdamOptimizer
from .flexflow_pybind11_internal import NetConfig, SingleDataLoader, Tensor, FFConfig, PerfMetrics, Op, Parameter

from .flexflow_pybind11_internal import FFModel as _FFModel

ff_tracing_id = 200

# -----------------------------------------------------------------------
# Op
# -----------------------------------------------------------------------
# @property
# def num_parameters():
#   return self.op.num_weights

def get_weight_tensor(self):
  parameter = self.get_parameter_by_id(0)
  return parameter
setattr(Op, "get_weight_tensor", get_weight_tensor)

def get_bias_tensor(self):
  parameter = self.get_parameter_by_id(1)
  return parameter
setattr(Op, "get_bias_tensor", get_bias_tensor)
    
# -----------------------------------------------------------------------
# Parameter
# -----------------------------------------------------------------------
# TODO: need to revisit it      
def get_weights(self, ffmodel):
  shape = self.dims
  np_array = np.empty(shape, dtype=np.float32)
  np_raw_ptr = np_array.__array_interface__['data']
  fflogger.debug("get weights raw_ptr: %s, %s, %s" %( str(np_raw_ptr[0]), hex(np_raw_ptr[0]), str(shape)))
  ret_val = self._get_weights(ffmodel, np_array)
  assert ret_val == True
  return np_array
setattr(Parameter, "get_weights", get_weights)
  
def set_weights(self, ffmodel, np_array):
  assert np_array.__array_interface__['strides'] == None, "Parameter set_weights, numpy array strides is not None"
  np_shape = np_array.shape
  num_dims = len(np_shape)
  assert num_dims == self.num_dims, "please check dims (%d == %d)" %(num_dims, self.parameter.num_dims)
  for i in range(0, num_dims):
    assert np_shape[i] == self.dims[i], "please check shape dim %d (%d == %d)" %(i, np_shape[i], self.dims[i])
  np_raw_ptr = np_array.__array_interface__['data']
  fflogger.debug("set weights raw_ptr: %s, %s, %s" %( str(np_raw_ptr[0]), hex(np_raw_ptr[0]), str(np_shape)))
  ret_val = self._set_weights(ffmodel, self.dims, np_array)
  assert ret_val == True, ret_val
setattr(Parameter, "set_weights", set_weights)

# -----------------------------------------------------------------------
# FFModel
# -----------------------------------------------------------------------

class FFModel(_FFModel):
  
  def __init__(self, ffconfig):
    super(FFModel, self).__init__(ffconfig)
    self._ffconfig = ffconfig
    global ff_tracing_id
    self._tracing_id = ff_tracing_id
    ff_tracing_id += 1
    
  def split(self, input, sizes, axis, name=None):
    if type(sizes) is list:
      split = sizes
    else:
      assert input.dims[axis] % sizes == 0, "Split dimension is not divisible"
      split = [input.dims[axis] // sizes for i in range(sizes)]
    return _FFModel.split(self, input, split, axis, name)
    
  def compile(self, optimizer=None, loss_type=None, metrics=None, comp_mode=CompMode.TRAINING):
    if optimizer != None:
      self.optimizer = optimizer
    self._compile(loss_type, metrics, comp_mode)
  
  def fit(self, x=None, y=None, batch_size=None, epochs=1):
    if (isinstance(x, list) == False):
      dataloaders = [x]
    else:
      dataloaders = x
    dataloaders.append(y)

    num_samples = y.num_samples
    batch_size = self._ffconfig.batch_size
    self._tracing_id += 1 # get a new tracing id
    for epoch in range(0,epochs):
      for d in dataloaders:
        d.reset()
      self.reset_metrics()
      iterations = num_samples / batch_size
      for iter in range(0, int(iterations)):
        for d in dataloaders:
          d.next_batch(self)
        self._ffconfig.begin_trace(self._tracing_id)
        self.forward()
        self.zero_gradients()
        self.backward()
        self.update()
        self._ffconfig.end_trace(self._tracing_id)
        
  def eval(self, x=None, y=None, batch_size=None):
    if (isinstance(x, list) == False):
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
    self._tracing_id += 1 # get a new tracing id
    for iter in range(0, int(iterations)):
      for d in dataloaders:
        d.next_batch(self)
      self._ffconfig.begin_trace(self._tracing_id)
      self.forward()
      self.compute_metrics()
      self._ffconfig.end_trace(self._tracing_id)
      
  def create_data_loader_test(self, batch_tensor, full_array):
      full_array_shape = full_array.shape
      num_samples = full_array_shape[0]
      num_dim = len(full_array_shape)
      if (full_array.dtype == "float32"):
        datatype = DataType.DT_FLOAT
      elif (full_array.dtype == "int32"):
        datatype = DataType.DT_INT32
      else:
        assert 0, "unsupported datatype"

      if (num_dim == 2):
        full_tensor = self.create_tensor([num_samples, full_array_shape[1]], datatype)
      elif (num_dim == 4):
        full_tensor = self.create_tensor([num_samples, full_array_shape[1], full_array_shape[2], full_array_shape[3]], datatype)
      else:
        assert 0, "unsupported dims"

      full_tensor.attach_numpy_array(self._ffconfig, full_array)
      dataloader = SingleDataLoader(self, batch_tensor, full_tensor, num_samples, datatype)
      full_tensor.detach_numpy_array(self._ffconfig)

      return dataloader
