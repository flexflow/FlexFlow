import flexflow.core as ff

from flexflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation

class Sequential(object):
  def __init__(self):
    self.ffconfig = ff.FFConfig()
    self.ffconfig.parse_args()
    print("Python API batchSize(%d) workersPerNodes(%d) numNodes(%d)" %(self.ffconfig.get_batch_size(), self.ffconfig.get_workers_per_node(), self.ffconfig.get_num_nodes()))
    self.ffmodel = ff.FFModel(self.ffconfig)
    
    self._layers = dict()
    self._nb_layers = 0
    self.output_tensor = 0
  
  def add(self, layer):
    self._layers[self._nb_layers] = layer
    self._nb_layers += 1
    
    if (isinstance(layer, Conv2D) == True):
      layer.handle = self.ffmodel.conv2d_v2(layer.name, layer.in_channels, layer.out_channels, layer.kernel_size, layer.kernel_size, layer.stride, layer.stride, layer.padding, layer.padding)
    elif (isinstance(layer, MaxPooling2D) == True):
      layer.handle = self.ffmodel.pool2d_v2(layer.name, layer.kernel_size, layer.kernel_size, layer.stride, layer.stride, layer.padding, layer.padding)
    elif (isinstance(layer, Flatten) == True):
      layer.handle = self.ffmodel.flat_v2(layer.name)
    elif (isinstance(layer, Dense) == True):
      layer.handle = self.ffmodel.dense_v2(layer.name, layer.in_channels, layer.out_channels, layer.activation)
    elif (isinstance(layer, Activation) == True):
      print("add softmax")
    else:
      assert 0, "unknow layer"
      
  def compile(self):
    self.ffoptimizer = ff.SGDOptimizer(self.ffmodel, 0.01)
    self.ffmodel.set_sgd_optimizer(self.ffoptimizer)
    
  def fit(self, input_tensor, label_tensor):
    for layer_id in self._layers:
      layer = self._layers[layer_id]
      if (layer_id == 0):
        self.output_tensor = layer.handle.init_inout(self.ffmodel, input_tensor);
      else:
        if (isinstance(layer, Activation) == True):
          assert layer_id == self._nb_layers-1, "softmax is not in the last layer"
          self.output_tensor = self.ffmodel.softmax("softmax", self.output_tensor, label_tensor)
        else:
          self.output_tensor = layer.handle.init_inout(self.ffmodel, self.output_tensor);
        
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
    
    