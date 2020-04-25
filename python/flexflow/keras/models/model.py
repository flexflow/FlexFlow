import flexflow.core as ff

from flexflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation

import builtins

def init_internal_model():
  builtins.internal_ffconfig = ff.FFConfig()
  builtins.internal_ffconfig.parse_args()
  builtins.internal_ffmodel = ff.FFModel(builtins.internal_ffconfig)

def delete_internal_model():
  del builtins.internal_ffconfig
  del builtins.internal_ffmodel

class Model(object):
  def __init__(self, input_tensor, output_tensor):
    self.ffconfig = ff.FFConfig()
    self.ffconfig.parse_args()
    print("Python API batchSize(%d) workersPerNodes(%d) numNodes(%d)" %(self.ffconfig.get_batch_size(), self.ffconfig.get_workers_per_node(), self.ffconfig.get_num_nodes()))
    self.ffmodel = ff.FFModel(self.ffconfig)
    
    self._layers = dict()
    self._nb_layers = 0
    self.input_tensor = input_tensor
    self.output_tensor = output_tensor
    
  def add(self, layer):
    self._layers[self._nb_layers] = layer
    layer.layer_id = self._nb_layers
    self._nb_layers += 1
    layer.add_to_model(self.ffmodel)
    
  def add_softmax(self, input_tensor, label_tensor):
    output_tensor = self.ffmodel.softmax("softmax", input_tensor, label_tensor)
    
  def compile(self):
    self.ffoptimizer = ff.SGDOptimizer(self.ffmodel, 0.01)
    self.ffmodel.set_sgd_optimizer(self.ffoptimizer)
    
  def fit(self, input_tensor, label_tensor):    
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