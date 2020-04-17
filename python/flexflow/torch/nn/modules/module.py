from collections import OrderedDict, namedtuple

from flexflow.core import *
from .conv import Conv2d
from .pooling import MaxPool2d
from .flatten import Flatten
from .linear import Linear
from .op import Op

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
    
    self._layers = dict()
    self._nb_layers = 0
    self._layer_inited = False
    
    self.ffoptimizer = SGDOptimizer(self.ffmodel, 0.01)
    self.ffmodel.set_sgd_optimizer(self.ffoptimizer)
  
  def __call__(self, input):
    output_tensor = self.forward(input)
    return output_tensor;
  
  def __setattr__(self, name, value):
    if (isinstance(value, Op) == True):
      value.set_flexflow_model(self.ffmodel)
      value.layer_id = self._nb_layers
      self._layers[name] = value
      self._nb_layers += 1
      
    if (isinstance(value, Conv2d) == True):
      value.handle = self.ffmodel.conv2d_v2(name, value.in_channels, value.out_channels, value.kernel_size[0], value.kernel_size[1], value.stride[0], value.stride[1], value.padding[0], value.padding[1])
    elif (isinstance(value, MaxPool2d) == True):
      value.handle = self.ffmodel.pool2d_v2(name, value.kernel_size, value.kernel_size, value.stride, value.stride, value.padding, value.padding)    
    elif (isinstance(value, Flatten) == True):
      value.handle = self.ffmodel.flat_v2(name)
    elif (isinstance(value, Linear) == True):
      value.handle = self.ffmodel.dense_v2(name, value.in_features, value.out_features, ActiMode.AC_MODE_RELU)
    else:
      #print("add others ", value)
      a=1
    super(Module, self).__setattr__(name, value)
    
  def init_inout(self, input):
    t = 0
    for layer in self._layers:
      layer_op = self._layers[layer]
      layer_id = layer_op.layer_id
      if (layer_id == 0):
        t = layer_op.handle.init_inout(self.ffmodel, input);
      else:
        t = layer_op.handle.init_inout(self.ffmodel, t);
    return t