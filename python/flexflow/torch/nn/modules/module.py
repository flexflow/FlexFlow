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
    
    dims = [self.ffconfig.get_batch_size(), 3, 229, 229]
    self.input = self.ffmodel.create_tensor_4d(dims, "", DataType.DT_FLOAT);
    self._layers = dict()
    self._nb_layers = 0
    
    dims_label = [self.ffconfig.get_batch_size(), 1]
    self.label = self.ffmodel.create_tensor_2d(dims_label, "", DataType.DT_INT32);
    self.dataloader = DataLoader(self.ffmodel, self.input, self.label, 1)
    
    self.ffoptimizer = SGDOptimizer(self.ffmodel, 0.01)
    self.ffmodel.set_sgd_optimizer(self.ffoptimizer)
  
  def __call__(self, input):
    
    self.input = self.ffmodel.softmax("softmax", self.input, self.label)
    self.ffmodel.init_layers()
    epochs = self.ffconfig.get_epochs()
  
    ts_start = self.ffconfig.get_current_time()
    for epoch in range(0,epochs):
      self.ffmodel.reset_metrics()
      iterations = 8192 / self.ffconfig.get_batch_size()
      for iter in range(0, int(iterations)):
        if (epoch > 0):
          self.ffconfig.begin_trace(111)
        self.ffmodel.forward()
        self.ffmodel.zero_gradients()
        self.ffmodel.backward()
        self.ffmodel.update()
        if (epoch > 0):
          self.ffconfig.end_trace(111)

    ts_end = self.ffconfig.get_current_time()
    run_time = 1e-6 * (ts_end - ts_start);
    print("epochs %d, ELAPSED TIME = %.4fs, THROUGHPUT = %.2f samples/s\n" %(epochs, run_time, 8192 * epochs / run_time));
    
    return self.forward(input)
  
  def __setattr__(self, name, value):
    if (isinstance(value, Op) == True):
      value.set_flexflow_model(self.ffmodel)
      value.set_layer_id(self._nb_layers)
      self._layers[name] = self._nb_layers
      self._nb_layers += 1
      
    if (isinstance(value, Conv2d) == True):
      self.input = self.ffmodel.conv2d(name, self.input, value.out_channels, value.kernel_size[0], value.kernel_size[1], value.stride[0], value.stride[1], value.padding[0], value.padding[1])
    elif (isinstance(value, MaxPool2d) == True):
      self.input = self.ffmodel.pool2d(name, self.input, value.kernel_size, value.kernel_size, value.stride, value.stride, value.padding, value.padding)    
    elif (isinstance(value, Flatten) == True):
      self.input = self.ffmodel.flat(name, self.input)
    elif (isinstance(value, Linear) == True):
      self.input = self.ffmodel.dense(name, self.input, value.out_features, ActiMode.AC_MODE_RELU)
    else:
      #print("add others ", value)
      a=1
    super(Module, self).__setattr__(name, value)