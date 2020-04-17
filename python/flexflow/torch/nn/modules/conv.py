from flexflow.core import *

from .utils import _single, _pair
from .op import Op

class _ConvNd(Op):
  def __init__(self, in_channels, out_channels, kernel_size, stride,
               padding, dilation, transposed, output_padding,
               groups, bias, padding_mode):
    super(_ConvNd, self).__init__()
    if in_channels % groups != 0:
      raise ValueError('in_channels must be divisible by groups')
    if out_channels % groups != 0:
      raise ValueError('out_channels must be divisible by groups')
    valid_padding_modes = {'zeros', 'reflect', 'replicate', 'circular'}
    if padding_mode not in valid_padding_modes:
      raise ValueError("padding_mode must be one of {}, but got padding_mode='{}'".format(
                       valid_padding_modes, padding_mode))
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel_size = kernel_size
    self.stride = stride
    self.padding = padding
    self.dilation = dilation
    self.transposed = transposed
    self.output_padding = output_padding
    self.groups = groups
    self.padding_mode = padding_mode
      
class Conv2d(_ConvNd):
  def __init__(self, in_channels, out_channels, kernel_size, stride=1,
               padding=0, dilation=1, groups=1,
               bias=True, padding_mode='zeros'):
    #print("create Conv2d")
    kernel_size = _pair(kernel_size)
    #print(kernel_size)
    stride = _pair(stride)
    padding = _pair(padding)
    dilation = _pair(dilation)
    super(Conv2d, self).__init__(
      in_channels, out_channels, kernel_size, stride, padding, dilation,
      False, _pair(0), groups, bias, padding_mode)
  
  def __call__(self, input):
    return self.forward(input)
    
  def forward(self, input):
    print("conv2d forward ", self.layer_id);
    input = self.handle.forward(self._ffmodel)
    return input
    