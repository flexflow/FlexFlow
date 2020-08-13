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
import random

from .base_layer import Layer
from .input_layer import Input
from flexflow.keras.models.tensor import Tensor
from flexflow.keras.initializers import Zeros, GlorotUniform, RandomUniform, RandomNormal, DefaultInitializer, Initializer

class Dense(Layer):
  __slots__ = ['in_channels', 'out_channels', 'activation', 'use_bias', \
               'kernel_initializer', 'bias_initializer']
  def __init__(self, units, input_shape=(0,), 
               activation=None, use_bias=True,
               kernel_initializer="glorot_uniform",
               bias_initializer="zeros",
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               **kwargs):
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
    
    super(Dense, self).__init__('dense', 'Dense', **kwargs) 
    
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
    self.out_channels = units
    self.use_bias = use_bias
    if (len(input_shape) == 2):
      self.in_channels = input_shape[1]
      self.input_shape = (input_shape[0], input_shape[1])
    elif (len(input_shape) == 1):
      self.in_channels = input_shape[0]
      self.input_shape = (0, input_shape[0])
    if (activation == None):
      self.activation = ff.ActiMode.AC_MODE_NONE
    elif(activation =="relu"):
      self.activation = ff.ActiMode.AC_MODE_RELU
    elif(activation =="sigmoid"):
      self.activation = ff.ActiMode.AC_MODE_SIGMOID
    else:
      assert 0, "activation is not supported"
    
  def verify_meta_data(self):
    assert self.input_shape != (0, 0), "input shape is wrong"
    assert self.output_shape != (0, 0), "output shape is wrong"
    assert self.in_channels != 0, " in channels is wrong"
    assert self.out_channels != 0, " out channels is wrong"
    
  def get_summary(self):
    summary = "%s%s\t\t%s%s\n"%(self._get_summary_name(), self.output_shape, self.input_shape, self._get_summary_connected_to())
    return summary
    
  def get_weights(self, ffmodel):
    return self._get_weights(ffmodel)
    
  def set_weights(self, ffmodel, kernel, bias):
    self._set_weights(ffmodel, kernel, bias)
  
  def __call__(self, input_tensor):
    return self._connect_layer_1_input_1_output(input_tensor)
    
  def _calculate_inout_shape(self, input_tensor):
    assert input_tensor.num_dims == 2, "[Dense]: shape of input tensor is wrong"
    input_b = input_tensor.batch_shape[0]
    in_dim = input_tensor.batch_shape[1]
    assert in_dim != 0, "wrong in_dim"
    if (self.in_channels != 0): # check if user input is correct
      assert self.in_channels == in_dim, "wrong input_w"
    self.output_shape = (input_b, self.out_channels)
    self.input_shape = (input_b, in_dim)
    self.in_channels = in_dim
    fflogger.debug("dense input %s, output %s" %( str(self.input_shape), str(self.output_shape)))
    
  def _verify_inout_tensor_shape(self, input_tensor, output_tensor):
    assert input_tensor.num_dims == 2, "[Dense]: check input tensor dims"
    assert input_tensor.batch_shape[1] == self.input_shape[1]
    assert output_tensor.num_dims == 2, "[Dense]: check output tensor dims"
    assert output_tensor.batch_shape[1] == self.output_shape[1]
    
  def _reset_layer(self):
    self.in_channels = 0
    
class Flatten(Layer):
  def __init__(self, data_format=None, **kwargs):
    if data_format != None:
      assert 0, "data_format is not supported"
    super(Flatten, self).__init__('flat', 'Flatten', **kwargs) 
    
  def verify_meta_data(self):
    assert self.input_shape != 0, "input shape is wrong"
    assert self.output_shape != (0, 0), "output shape is wrong"
    
  def get_summary(self):
    summary = "%s%s\t\t%s%s\n"%(self._get_summary_name(), self.output_shape, self.input_shape, self._get_summary_connected_to())
    return summary
    
  def __call__(self, input_tensor):    
    return self._connect_layer_1_input_1_output(input_tensor)
    
  def _calculate_inout_shape(self, input_tensor):
    input_shape = input_tensor.batch_shape
    self.input_shape = input_shape
    flat_size = 1
    for i in range(1, len(input_shape)):
      flat_size *= input_shape[i]
    self.output_shape = (input_shape[0], flat_size)
    fflogger.debug("flat input %s, output %s" %( str(self.input_shape), str(self.output_shape)))
    
  def _verify_inout_tensor_shape(self, input_tensor, output_tensor):
    assert input_tensor.num_dims == len(self.input_shape), "[Flatten]: check input tensor dims"
    for i in range (1, input_tensor.num_dims):
      assert input_tensor.batch_shape[i] == self.input_shape[i]
    assert output_tensor.num_dims == 2, "[Flatten]: check output tensor dims"
    assert output_tensor.batch_shape[1] == self.output_shape[1]
    
  def _reset_layer(self):
    pass
    
class Embedding(Layer):
  def __init__(self, 
               input_dim,
               output_dim,
               embeddings_initializer="uniform",
               embeddings_regularizer=None,
               activity_regularizer=None,
               embeddings_constraint=None,
               mask_zero=False,
               input_length=None,
               **kwargs):
    self.input_dim = input_dim
    self.out_channels = output_dim
    self.input_length = input_length
    
    if embeddings_initializer == "uniform":
      self.embeddings_initializer = RandomUniform(random.randint(0,1024), -0.05, 0.05)
      
    super(Embedding, self).__init__("embedding", "Embedding", **kwargs) 
      
  def verify_meta_data(self):
    pass
    
  def get_summary(self):
    summary = "%s%s\t\t%s%s\n"%(self._get_summary_name(), self.output_shape, self.input_shape, self._get_summary_connected_to())
    return summary
    
  def __call__(self, input_tensor):
    return self._connect_layer_1_input_1_output(input_tensor)
    
  def _calculate_inout_shape(self, input_tensor):
    assert input_tensor.num_dims == 2, "[Embedding]: shape of input tensor is wrong"
    input_b = input_tensor.batch_shape[0]
    in_dim = input_tensor.batch_shape[1]
    assert in_dim != 0, "wrong in_dim"
    assert self.input_length == in_dim, "wrong input_w"
    self.output_shape = (input_b, self.out_channels)
    self.input_shape = (input_b, self.input_length)
    fflogger.debug("embedding input %s, output %s" %( str(self.input_shape), str(self.output_shape)))
    
  def _verify_inout_tensor_shape(self, input_tensor, output_tensor):
    assert input_tensor.num_dims == 2, "[Embedding]: check input tensor dims"
    assert input_tensor.batch_shape[1] == self.input_shape[1]
    assert output_tensor.num_dims == 2, "[Embedding]: check output tensor dims"
    assert output_tensor.batch_shape[1] == self.output_shape[1]
    
  def _reset_layer(self):
    pass
    
class Activation(Layer):
  def __init__(self, activation=None, **kwargs):
    
    if (activation == 'softmax') or (activation == 'relu') or (activation == 'sigmoid') or (activation == 'tanh') or (activation == 'elu'):
      self.activation = activation
    else:
      assert 0, '[Activation]: unsupported activation'
      
    super(Activation, self).__init__(self.activation, 'Activation', **kwargs) 
      
  def verify_meta_data(self):
    pass
    
  def get_summary(self):
    summary = "%s%s\t\t%s%s\n"%(self._get_summary_name(), self.output_shape, self.input_shape, self._get_summary_connected_to())
    return summary
    
  def __call__(self, input_tensor):
    return self._connect_layer_1_input_1_output(input_tensor)
    
  def _calculate_inout_shape(self, input_tensor):
    self.input_shape = input_tensor.batch_shape
    self.output_shape = input_tensor.batch_shape
    
  def _verify_inout_tensor_shape(self, input_tensor, output_tensor):
    pass
    
  def _reset_layer(self):
    pass
    
class Dropout(Layer):
  def __init__(self, rate, noise_shape=None, seed=None, **kwargs):
    if noise_shape != None:
      assert 0, "noise_shape is not supported"
    self.rate = rate
    self.noise_shape = noise_shape
    if seed == None:
      _seed = 0
    self.seed = _seed
      
    super(Dropout, self).__init__('dropout', 'Dropout', **kwargs) 
      
  def verify_meta_data(self):
    pass
    
  def get_summary(self):
    summary = "%s%s\t\t%s%s\n"%(self._get_summary_name(), self.output_shape, self.input_shape, self._get_summary_connected_to())
    return summary
    
  def __call__(self, input_tensor):
    return self._connect_layer_1_input_1_output(input_tensor)
    
  def _calculate_inout_shape(self, input_tensor):
    self.input_shape = input_tensor.batch_shape
    self.output_shape = input_tensor.batch_shape
    
  def _verify_inout_tensor_shape(self, input_tensor, output_tensor):
    pass
    
  def _reset_layer(self):
    pass