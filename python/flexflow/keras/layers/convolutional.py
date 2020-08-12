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
from flexflow.keras.initializers import Zeros, GlorotUniform, RandomUniform, RandomNormal, DefaultInitializer, Initializer

class Conv2D(Layer):
  __slots__ = ['in_channels', 'out_channels', 'kernel_size', 'stride', \
               'padding', 'activation', 'use_bias', 'kernel_initializer', \
               'bias_initializer']
  def __init__(self, 
               filters, 
               input_shape=(0,), 
               kernel_size=0, 
               strides=(1, 1), 
               padding="valid", 
               data_format=None, 
               dilation_rate=(1, 1),
               groups=1, 
               activation=None, 
               use_bias=True, 
               kernel_initializer='glorot_uniform', 
               bias_initializer='zeros', 
               kernel_regularizer=None, 
               bias_regularizer=None, 
               activity_regularizer=None, 
               kernel_constraint=None, 
               bias_constraint=None, 
               **kwargs):
    if data_format == 'channels_last':
      assert 0, "data_format channels_last is not supported"
    if dilation_rate != (1,1):
      assert 0, "dilation_rate is not supported"
    if groups != 1:
      assert 0, "groups is not supported"
    if kernel_regularizer != None:
      assert 0, "kernel_regularizer is not supported"
    if bias_regularizer != None:
      assert 0, "bias_regularizer is not supported"
    if activity_regularizer != None:
      assert 0, "activity_regularizer is not supported"
    if kernel_constraint != None:
      assert 0, "kernel_constraint is not supported"
    if bias_constraint != None:
      assert 0, "bias_constraint is not supported"
    
    super(Conv2D, self).__init__("conv2d", "Conv2D", **kwargs) 
    
    if kernel_initializer == "glorot_uniform":
      self.kernel_initializer = DefaultInitializer()
    elif isinstance(kernel_initializer, Initializer) == True:
      self.kernel_initializer = kernel_initializer
    else:
      assert 0, "[Dense]: unknown kernel_initializer"
      
    if bias_initializer == "zeros":
      self.bias_initializer = DefaultInitializer()
    elif isinstance(bias_initializer, Initializer) == True:
      self.bias_initializer = bias_initializer
    else:
      assert 0, "[Dense]: unknown bias_initializer"
    
    self.in_channels = 0
    self.out_channels = filters
    assert len(kernel_size)==2, "wrong dim of kernel_size"
    self.kernel_size = kernel_size
    assert len(strides)==2, "wrong dim of stride"
    self.stride = strides
    if (padding == "valid"):
      self.padding = (0, 0)
    elif (padding == "same"):
      self.padding = "same"
    elif (isinstance(padding, list) or isinstance(padding, tuple)):
      assert len(padding)==2, "[Conv2D]: wrong dim of padding"
      self.padding = tuple(padding)
    else:
      assert 0, "[Conv2D]: check padding"
    if (activation == None):
      self.activation = ff.ActiMode.AC_MODE_NONE
    elif(activation =="relu"):
      self.activation = ff.ActiMode.AC_MODE_RELU
    else:
      assert 0, "activation is not supported"
    if (len(input_shape) == 4):
      self.in_channels = input_shape[1]
      self.input_shape = (input_shape[0], input_shape[1], input_shape[2], input_shape[3])
    elif (len(input_shape) == 3):
      self.in_channels = input_shape[0]
      self.input_shape = (0, input_shape[0], input_shape[1], input_shape[2])
    self.use_bias = use_bias
  
  def verify_meta_data(self):
    assert self.input_shape != (0, 0, 0, 0), "[Conv2D]: input shape is wrong"
    assert self.output_shape != (0, 0, 0, 0), "[Conv2D]: output shape is wrong"
    assert self.in_channels != 0, "[Conv2D]: in channels is wrong"
    assert self.out_channels != 0, "[Conv2D]: out channels is wrong"
    
  def get_summary(self):
    summary = "%s%s\t\t%s\t%s\n"%(self._get_summary_name(), self.output_shape, self.input_shape, self._get_summary_connected_to())
    return summary
    
  def get_weights(self, ffmodel):
    return self._get_weights(ffmodel)
    
  def set_weights(self, ffmodel, kernel, bias):
    self._set_weights(ffmodel, kernel, bias)
    
  def __call__(self, input_tensor):
    return self._connect_layer_1_input_1_output(input_tensor)
    
  def _calculate_inout_shape(self, input_tensor):
    assert input_tensor.num_dims == 4, "[Conv2D]: shape of input tensor is wrong"
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
      fflogger.debug("conv2d same padding %s" %(str(self.padding)))
    
    self.input_shape = (input_b, input_d, input_w, input_h)
    self.in_channels = input_d
    output_h = 1 + math.floor((input_h + 2 * self.padding[0] - self.kernel_size[0]) / self.stride[0])
    output_w = 1 + math.floor((input_w + 2 * self.padding[1] - self.kernel_size[1]) / self.stride[1])
    output_d = self.out_channels
    self.output_shape = (input_b, output_d, output_h, output_w)
    fflogger.debug("conv2d input %s, output %s" %( str(self.input_shape), str(self.output_shape)))
    
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