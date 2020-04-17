from flexflow.core import *

from .op import Op

class _MaxPoolNd(Op):
  def __init__(self, kernel_size, stride=None, padding=0, dilation=1,
               return_indices=False, ceil_mode=False):
    super(_MaxPoolNd, self).__init__()
    self.kernel_size = kernel_size
    self.stride = stride or kernel_size
    self.padding = padding
    self.dilation = dilation
    self.return_indices = return_indices
    self.ceil_mode = ceil_mode
    
class MaxPool2d(_MaxPoolNd):
  def __init__(self, kernel_size, stride=None, padding=0, dilation=1,
               return_indices=False, ceil_mode=False):
    print("create MaxPool2d")
    super(MaxPool2d, self).__init__(
      kernel_size, stride, padding, dilation,
      return_indices, ceil_mode)
    
  def __call__(self, input):
    return self.forward(input)
    
  def forward(self, input):
    print("maxpool2d forward ", self.layer_id);
    self.handle.forward(self._ffmodel)
    return input
    
class AvgPool2d(object):
  def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False,
               count_include_pad=True, divisor_override=None):
    print("create AvgPool2d")
   # super(AvgPool2d, self).__init__()
    self.kernel_size = kernel_size
    self.stride = stride or kernel_size
    self.padding = padding
    self.ceil_mode = ceil_mode
    self.count_include_pad = count_include_pad
    self.divisor_override = divisor_override
    