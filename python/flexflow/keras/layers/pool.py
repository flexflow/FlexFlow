import flexflow.core as ff
import math

from .base_layer import Layer
from flexflow.keras.models.input_layer import Tensor, Input

class MaxPooling2D(Layer):
  def __init__(self, pool_size, strides, padding="valid", name="pool2d"):
    super(MaxPooling2D, self).__init__(name, "MaxPooling2D") 
    
    self.input_shape = 0
    self.output_shape = 0
    self.out_channels = 0
    self.in_channels = 0
    assert len(pool_size)==2, "wrong dim of pool_size"
    self.kernel_size = pool_size
    assert len(strides)==2, "wrong dim of strides"
    self.stride = strides
    if (padding == "valid"):
      self.padding = (0, 0)
    elif (padding == "same"):
      self.padding = (0, 0)
    elif (isinstance(padding, list) or isinstance(padding, tuple)):
      assert len(padding)==2, "[MaxPooling2D]: wrong dim of padding"
      self.padding = tuple(padding)
    else:
      assert 0, "[MaxPooling2D]: check padding"
    
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
    input_w = input_tensor.batch_shape[2]
    input_h = input_tensor.batch_shape[3]
    assert input_w != 0, "wrong input_w"
    assert input_h != 0, "wrong input_h"
    assert input_d != 0, "wrong input_d"
    self.input_shape = (input_b, input_d, input_w, input_h)
    self.in_channels = input_d
    self.out_channels = input_d
    output_w = 1 + math.floor((input_w + 2 * self.padding[0] - self.kernel_size[0]) / self.stride[0])
    output_h = 1 + math.floor((input_h + 2 * self.padding[1] - self.kernel_size[1]) / self.stride[1])
    output_d = self.out_channels
    self.output_shape = (input_b, output_d, output_w, output_h)
    print("pool2d input ", self.input_shape)
    print("pool2d output ", self.output_shape)
    
  def _verify_inout_tensor_shape(self, input_tensor, output_tensor):
    assert input_tensor.num_dims == 4, "[Conv2D]: check input tensor dims"
    assert input_tensor.batch_shape[1] == self.input_shape[1]
    assert input_tensor.batch_shape[2] == self.input_shape[2]
    assert input_tensor.batch_shape[3] == self.input_shape[3]
    assert output_tensor.num_dims == 4, "[Conv2D]: check output tensor dims"
    assert output_tensor.batch_shape[1] == self.output_shape[1]
    assert output_tensor.batch_shape[2] == self.output_shape[2]
    assert output_tensor.batch_shape[3] == self.output_shape[3]