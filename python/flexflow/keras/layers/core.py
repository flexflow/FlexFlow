import flexflow.core as ff

from .base_layer import Layer
from .input_layer import Input
from flexflow.keras.models.tensor import Tensor

class Dense(Layer):
  __slots__ = ['in_channels', 'out_channels', 'activation', 'use_bias']
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
    if kernel_initializer != "glorot_uniform":
      assert 0, "kernel_initializer is not supported"
    if bias_initializer != "zeros":
      assert 0, "bias_initializer is not supported"
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
    
    super(Dense, self).__init__("dense", "Dense" ,**kwargs) 
    
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
    print("dense input ", self.input_shape)
    print("dense output ", self.output_shape)
    
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
    super(Flatten, self).__init__("flat", "Flatten", **kwargs) 
    
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
    print("flat input ", self.input_shape)
    print("flat output ", self.output_shape)
    
  def _verify_inout_tensor_shape(self, input_tensor, output_tensor):
    assert input_tensor.num_dims == len(self.input_shape), "[Flatten]: check input tensor dims"
    for i in range (1, input_tensor.num_dims):
      assert input_tensor.batch_shape[i] == self.input_shape[i]
    assert output_tensor.num_dims == 2, "[Flatten]: check output tensor dims"
    assert output_tensor.batch_shape[1] == self.output_shape[1]
    
  def _reset_layer(self):
    pass
    
class Activation(Layer):
  def __init__(self, activation, **kwargs):
    
    if (activation == "softmax"):
      self.activation = "Softmax"
      
    super(Activation, self).__init__("activation", self.activation, **kwargs) 
      
  def verify_meta_data(self):
    assert self.activation == "Softmax", "type is wrong"
    
  def get_summary(self):
    summary = "%s%s\n"%(self._get_summary_name(), self._get_summary_connected_to())
    return summary
    
  def __call__(self, input_tensor):
    return self._connect_layer_1_input_1_output(input_tensor)
    
  def _calculate_inout_shape(self, input_tensor):
    assert input_tensor.num_dims == 2, "[Activation]: shape of input tensor is wrong"
    self.input_shape = input_tensor.batch_shape
    self.output_shape = input_tensor.batch_shape
    
  def _verify_inout_tensor_shape(self, input_tensor, output_tensor):
    pass
    
  def _reset_layer(self):
    pass