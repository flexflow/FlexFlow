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

import flexflow.core as ff
from flexflow.core.flexflow_logger import fflogger

from .base_layer import Layer
from .input_layer import Input
from flexflow.keras.models.tensor import Tensor

class _Merge(Layer):
  def __init__(self, default_name, layer_type):
    super(_Merge, self).__init__(default_name, layer_type) 
  
  def verify_meta_data(self):
   pass
    
  def get_summary(self):
    summary = "%s%s%s\n"%(self._get_summary_name(), self.output_shape, self._get_summary_connected_to())
    return summary
    
  def __call__(self, input_tensors):
    return self._connect_layer_n_input_1_output(input_tensors)
    
  def _verify_inout_tensor_shape(self, input_tensors, output_tensor):
    assert self.__check_duplications(input_tensors) == False, "[Merge]: dunpicated input_tensors is not supported"
    for input_tensor in input_tensors:
      assert input_tensor.num_dims == len(self.input_shape), "[Merge]: check input tensor dims"
      for i in range (1, input_tensor.num_dims):
        if isinstance(self, Concatenate) and self.axis == i:
          continue
        # Merge functions other than Concatenate allow broadcasting
        assert (input_tensor.batch_shape[i] == self.input_shape[i] or
                input_tensor.batch_shape[i] == 1 or
                self.input_shape[i] == 1
               ), ('Incompatible shapes for broadcasting: '
                   f'{input_tensor.batch_shape} and {self.input_shape}')
    assert output_tensor.num_dims == len(self.output_shape), "[Merge]: check output tensor dims"
    for i in range (1, output_tensor.num_dims):
      assert output_tensor.batch_shape[i] == self.output_shape[i]
      
  def _reset_layer(self):
    pass
    
  def __check_duplications(self, input_tensors):
    if len(input_tensors) == len(set(input_tensors)):
      return False
    else:
      return True
    
def concatenate(input_tensors, _axis=1):
  return Concatenate(axis=_axis)(input_tensors)
    
class Concatenate(_Merge):
  __slots__ = ['axis']
  def __init__(self, axis, **kwargs):
    super(Concatenate, self).__init__("concatenate", "Concatenate", **kwargs) 
    
    self.axis = axis
    
  def _calculate_inout_shape(self, input_tensors):
    if (input_tensors[0].num_dims == 2):
      output_shape = [input_tensors[0].batch_shape[0], 0]
      for input_tensor in input_tensors:
        output_shape[self.axis] += input_tensor.batch_shape[self.axis]
      self.output_shape = (output_shape[0], output_shape[1])
    elif (input_tensors[0].num_dims == 3):
      output_shape = [input_tensors[0].batch_shape[0], input_tensors[0].batch_shape[1], input_tensors[0].batch_shape[2]]
      for input_tensor in input_tensors[1:]:
        output_shape[self.axis] += input_tensor.batch_shape[self.axis]
      self.output_shape = (output_shape[0], output_shape[1], output_shape[2])
    elif (input_tensors[0].num_dims == 4):
      output_shape = [input_tensors[0].batch_shape[0], 0, input_tensors[0].batch_shape[2], input_tensors[0].batch_shape[3]]
      for input_tensor in input_tensors:
        output_shape[self.axis] += input_tensor.batch_shape[self.axis]
      self.output_shape = (output_shape[0], output_shape[1], output_shape[2], output_shape[3])
    else:
      assert 0, "un-supported dims"
    fflogger.debug("concat output %s" %( str(self.output_shape)))
    self.input_shape = input_tensors[0].batch_shape

def add(input_tensors):
  return Add()(input_tensors)
    
class Add(_Merge):
  def __init__(self, **kwargs):
    super(Add, self).__init__("add", "Add", **kwargs) 
    
  def _calculate_inout_shape(self, input_tensors):    
    assert len(input_tensors) == 2, "check input_tensors"   
    self.input_shape = input_tensors[0].batch_shape
    self.output_shape = list(input_tensors[0].batch_shape)
    for i, d in enumerate(input_tensors[1].batch_shape):
      if self.output_shape[i] != d:
        if self.output_shape[i] == 1 or d == 1:
          self.output_shape[i] *= d
        else:
          raise AssertionError(
            f"Tensor with shape {input_tensors[0].batch_shape} and "
            f"{input_tensors[1].batch_shape} cannot be added")
    self.output_shape = tuple(self.output_shape)
    fflogger.debug("add output %s" %( str(self.output_shape)))
    
def subtract(input_tensors):
  return Subtract()(input_tensors)
    
class Subtract(_Merge):
  def __init__(self, **kwargs):
    super(Subtract, self).__init__("subtract", "Subtract", **kwargs) 
    
  def _calculate_inout_shape(self, input_tensors): 
    assert len(input_tensors) == 2, "check input_tensors"   
    self.input_shape = input_tensors[0].batch_shape
    self.output_shape = list(input_tensors[0].batch_shape)
    for i, d in enumerate(input_tensors[1].batch_shape):
      if self.output_shape[i] != d:
        if self.output_shape[i] == 1 or d == 1:
          self.output_shape[i] *= d
        else:
          raise AssertionError(
            f"Tensor with shape {input_tensors[0].batch_shape} and "
            f"{input_tensors[1].batch_shape} cannot be subtracted")
    self.output_shape = tuple(self.output_shape)
    fflogger.debug("subtract output %s" %( str(self.output_shape)))

def multiply(input_tensors):
  return Multiply()(input_tensors)
    
class Multiply(_Merge):
  def __init__(self, **kwargs):
    super(Multiply, self).__init__("multiply", "Multiply", **kwargs) 
    
  def _calculate_inout_shape(self, input_tensors): 
    assert len(input_tensors) == 2, "check input_tensors"   
    self.input_shape = input_tensors[0].batch_shape
    self.output_shape = list(input_tensors[0].batch_shape)
    for i, d in enumerate(input_tensors[1].batch_shape):
      if self.output_shape[i] != d:
        if self.output_shape[i] == 1 or d == 1:
          self.output_shape[i] *= d
        else:
          raise AssertionError(
            f"Tensor with shape {input_tensors[0].batch_shape} and "
            f"{input_tensors[1].batch_shape} cannot be multiplied")
    self.output_shape = tuple(self.output_shape)
    fflogger.debug("multiply output %s" %( str(self.output_shape)))

class Maximum(_Merge):
  def __init__(self, **kwargs):
    super(Maximum, self).__init__("maximum", "Maximum", **kwargs) 
    
  def _calculate_inout_shape(self, input_tensors): 
    assert len(input_tensors) == 2, "check input_tensors"   
    self.input_shape = input_tensors[0].batch_shape
    self.output_shape = input_tensors[0].batch_shape
    fflogger.debug("maximum output %s" %( str(self.output_shape)))

class Minimum(_Merge):
  def __init__(self, **kwargs):
    super(Minimum, self).__init__("minimum", "Minimum", **kwargs) 
    
  def _calculate_inout_shape(self, input_tensors): 
    assert len(input_tensors) == 2, "check input_tensors"   
    self.input_shape = input_tensors[0].batch_shape
    self.output_shape = input_tensors[0].batch_shape
    fflogger.debug("minimum output %s" %( str(self.output_shape)))
