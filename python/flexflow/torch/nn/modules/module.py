from collections import OrderedDict, namedtuple

from flexflow.core import *
from .conv import Conv2d
from .pooling import MaxPool2d
from .flatten import Flatten

class _IncompatibleKeys(namedtuple('IncompatibleKeys', ['missing_keys', 'unexpected_keys'])):
  def __repr__(self):
    if not self.missing_keys and not self.unexpected_keys:
        return '<All keys matched successfully>'
    return super(_IncompatibleKeys, self).__repr__()

  __str__ = __repr__

class Module(object):
  def __init__(self):
    self.ffconfig = FFConfig()
    self.ffconfig.parse_args()
    print("Python API batchSize(%d) workersPerNodes(%d) numNodes(%d)" %(self.ffconfig.get_batch_size(), self.ffconfig.get_workers_per_node(), self.ffconfig.get_num_nodes()))
    self.ffmodel = FFModel(self.ffconfig)
    
    dims = [self.ffconfig.get_batch_size(), 3, 229, 229]
    self.input = self.ffmodel.create_tensor_4d(dims, "", DataType.DT_FLOAT);
  
  def __setattr__(self, name, value):
    super(Module, self).__setattr__(name, value)
    if (isinstance(value, FFModel) == True):
      print("!constructor add FFModel")
    elif (isinstance(value, FFConfig) == True):
      print("!constructor add FFConfig")
    else:
      if (isinstance(value, Conv2d) == True):
        self.input = self.ffmodel.conv2d("conv1", self.input, value.out_channels, value.kernel_size[0], value.kernel_size[1], value.stride[0], value.stride[1], value.padding[0], value.padding[1])
      elif (isinstance(value, MaxPool2d) == True):
        self.input = self.ffmodel.pool2d("pool1", self.input, value.kernel_size, value.kernel_size, value.stride, value.stride, value.padding, value.padding)    
      elif (isinstance(value, Flatten) == True):
        self.input = self.ffmodel.flat("flat1", self.input)