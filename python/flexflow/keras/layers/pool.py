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

import flexflow.core as ff
from flexflow.core.flexflow_logger import fflogger
import math

from .base_layer import Layer
from .input_layer import Input
from flexflow.keras.models.tensor import Tensor

class Pooling2D(Layer):
  __slots__ = ['in_channels', 'out_channels', 'kernel_size', 'stride', \
               'padding', 'pool_type']
  def __init__(self, pool_size, strides, padding="valid", default_name="pool2d", pool_type=ff.PoolType.POOL_MAX, layer_type="MaxPooling2D", **kwargs):
    super(Pooling2D, self).__init__(default_name, layer_type, **kwargs) 
    
    self.in_channels = 0
    self.out_channels = 0
    assert len(pool_size)==2, "wrong dim of pool_size"
    self.kernel_size = pool_size
    assert len(strides)==2, "wrong dim of strides"
    self.stride = strides
    if (padding == "valid"):
      self.padding = (0, 0)
    elif (padding == "same"):
      self.padding = "same"
    elif (isinstance(padding, list) or isinstance(padding, tuple)):
      assert len(padding)==2, "[Pooling2D]: wrong dim of padding"
      self.padding = tuple(padding)
    else:
      assert 0, "[Pooling2D]: check padding"
    self.pool_type = pool_type
    
  def verify_meta_data(self):
    assert self.input_shape != (0, 0, 0, 0), "input shape is wrong"
    assert self.output_shape != (0, 0, 0, 0), "output shape is wrong"
    assert self.in_channels != 0, " in channels is wrong"
    assert self.out_channels != 0, " out channels is wrong"
    
  def get_summary(self):
    summary = "%s%s\t\t%s%s\n"%(self._get_summary_name(), self.output_shape, self.input_shape, self._get_summary_connected_to())
    return summary
    
  def __call__(self, input_tensor):
    return self._connect_layer_1_input_1_output(input_tensor)
    
  def _calculate_inout_shape(self, input_tensor):
    assert input_tensor.num_dims == 4, "[MaxPooling2D]: shape of input tensor is wrong"
    input_b = input_tensor.batch_shape[0]
    input_d = input_tensor.batch_shape[1]
    input_h = input_tensor.batch_shape[2]
    input_w = input_tensor.batch_shape[3]
    assert input_h != 0, "wrong input_h"
    assert input_w != 0, "wrong input_w"
    assert input_d != 0, "wrong input_d"
    
    #calculate padding for same
    if (self.padding == 'same'):
      if (input_h % self.stride[0] == 0):
        padding_h = max(self.kernel_size[0] - self.stride[0], 0)
      else:
        padding_h = max(self.kernel_size[0] - (input_h % self.stride[0]), 0)
      if (input_w % self.stride[1] == 0):
        padding_w = max(self.kernel_size[1] - self.stride[1], 0)
      else:
        padding_w = max(self.kernel_size[1] - (input_w % self.stride[1]), 0)
      self.padding = (padding_h//2, padding_w//2)
      fflogger.debug("pool2d same padding %s" %( str(self.padding)))
      
    self.input_shape = (input_b, input_d, input_w, input_h)
    self.in_channels = input_d
    self.out_channels = input_d
    output_h = 1 + math.floor((input_h + 2 * self.padding[0] - self.kernel_size[0]) / self.stride[0])
    output_w = 1 + math.floor((input_w + 2 * self.padding[1] - self.kernel_size[1]) / self.stride[1])
    output_d = self.out_channels
    self.output_shape = (input_b, output_d, output_h, output_w)
    fflogger.debug("pool2d input %s, output %s" %( str(self.input_shape), str(self.output_shape)))
    
  def _verify_inout_tensor_shape(self, input_tensor, output_tensor):
    assert input_tensor.num_dims == 4, "[Conv2D]: check input tensor dims"
    assert input_tensor.batch_shape[1] == self.input_shape[1]
    assert input_tensor.batch_shape[2] == self.input_shape[2]
    assert input_tensor.batch_shape[3] == self.input_shape[3]
    assert output_tensor.num_dims == 4, "[Conv2D]: check output tensor dims"
    assert output_tensor.batch_shape[1] == self.output_shape[1]
    assert output_tensor.batch_shape[2] == self.output_shape[2]
    assert output_tensor.batch_shape[3] == self.output_shape[3]
  
  def _reset_layer(self):
    self.in_channels = 0
    
class MaxPooling2D(Pooling2D):
  def __init__(self, pool_size=(2, 2), strides=None, padding="valid", data_format=None, **kwargs):
    if data_format == 'channels_last':
      assert 0, "data_format channels_last is not supported"
    super(MaxPooling2D, self).__init__(pool_size, strides, padding, "maxpool2d", ff.PoolType.POOL_MAX, "MaxPooling2D", **kwargs) 
    
class AveragePooling2D(Pooling2D):
  def __init__(self, pool_size=(2, 2), strides=None, padding="valid", data_format=None, **kwargs):
    if data_format == 'channels_last':
      assert 0, "data_format channels_last is not supported"
    super(AveragePooling2D, self).__init__(pool_size, strides, padding, "averagepool2d", ff.PoolType.POOL_AVG, "AveragePooling2D", **kwargs) 