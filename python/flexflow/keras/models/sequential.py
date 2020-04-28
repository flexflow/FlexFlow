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
    self.input_tensor = 0
    self.output_tensor = 0
    self.use_v2 = False
    
  def _create_layer_and_init_inout(self, input_tensor, label_tensor, verify_inout_shape=True):
    int_t = 0
    out_t = 0
    self.input_tensor = input_tensor
    for layer_id in self._layers:
      layer = self._layers[layer_id]
      assert layer.layer_id == layer_id, "wrong layer id"
      if (layer_id == 0):
        in_t = input_tensor
      else:
        in_t = out_t
        
      if (isinstance(layer, Conv2D) == True):
        out_t = self.ffmodel.conv2d(layer.name, in_t, layer.out_channels, layer.kernel_size[0], layer.kernel_size[1], layer.stride[0], layer.stride[1], layer.padding[0], layer.padding[1], layer.activation, layer.use_bias)
      elif (isinstance(layer, MaxPooling2D) == True):
        out_t = self.ffmodel.pool2d(layer.name, in_t, layer.kernel_size[0], layer.kernel_size[1], layer.stride[0], layer.stride[1], layer.padding[0], layer.padding[1])
      elif (isinstance(layer, Flatten) == True):
        out_t = self.ffmodel.flat(layer.name, in_t)
      elif (isinstance(layer, Dense) == True):
        out_t = self.ffmodel.dense(layer.name, in_t, layer.out_channels, layer.activation)
      elif (isinstance(layer, Activation) == True):
        assert layer_id == self._nb_layers-1, "softmax is not in the last layer"
        out_t = self.ffmodel.softmax("softmax", in_t, label_tensor)
        
      if (verify_inout_shape == True):
        layer.verify_inout_shape(in_t, out_t) 
      layer.handle = self.ffmodel.get_layer_by_id(layer_id)
      print(layer.handle)
    self.output_tensor = out_t
    
  def _init_inout(self, input_tensor, label_tensor):
    self.input_tensor = input_tensor
    for layer_id in self._layers:
      layer = self._layers[layer_id]
      if (layer_id == 0):
        self.output_tensor = layer.handle.init_inout(self.ffmodel, input_tensor);
      else:
        if (isinstance(layer, Activation) == True):
          assert layer_id == self._nb_layers-1, "softmax is not in the last layer"
          self.output_tensor = self.ffmodel.softmax("softmax", self.output_tensor, label_tensor)
          layer.handle = self.ffmodel.get_layer_by_id(layer_id)
        else:
          self.output_tensor = layer.handle.init_inout(self.ffmodel, self.output_tensor);
    
  def add(self, layer):
    self._layers[self._nb_layers] = layer
    layer.layer_id = self._nb_layers
    self._nb_layers += 1
    
    if (isinstance(layer, Conv2D) == True):
      if (layer.layer_id > 0):
        prev_layer = self._layers[layer.layer_id-1]
        layer.calculate_inout_shape(prev_layer.output_shape[1], prev_layer.output_shape[2], prev_layer.output_shape[3], prev_layer.output_shape[0])
    elif (isinstance(layer, MaxPooling2D) == True):
      assert layer.layer_id != 0, "maxpool2d can not be the 1st layer"
      prev_layer = self._layers[layer.layer_id-1]
      layer.calculate_inout_shape(prev_layer.output_shape[1], prev_layer.output_shape[2], prev_layer.output_shape[3], prev_layer.output_shape[0])
    elif (isinstance(layer, Flatten) == True):
      assert layer.layer_id != 0, "flatten can not be the 1st layer"
      prev_layer = self._layers[layer.layer_id-1]
      layer.calculate_inout_shape(prev_layer.output_shape)
    elif (isinstance(layer, Dense) == True):
      if (layer.layer_id > 0):
        prev_layer = self._layers[layer.layer_id-1]
        layer.calculate_inout_shape(prev_layer.output_shape[1])
  
  def add_v2(self, layer):
    self.use_v2 = True
    self._layers[self._nb_layers] = layer
    layer.layer_id = self._nb_layers
    self._nb_layers += 1
    
    if (isinstance(layer, Conv2D) == True):
      if (layer.layer_id > 0):
        prev_layer = self._layers[layer.layer_id-1]
        layer.calculate_inout_shape(prev_layer.output_shape[1], prev_layer.output_shape[2], prev_layer.output_shape[3], prev_layer.output_shape[0])
    elif (isinstance(layer, MaxPooling2D) == True):
      assert layer.layer_id != 0, "maxpool2d can not be the 1st layer"
      prev_layer = self._layers[layer.layer_id-1]
      layer.calculate_inout_shape(prev_layer.output_shape[1], prev_layer.output_shape[2], prev_layer.output_shape[3], prev_layer.output_shape[0])
    elif (isinstance(layer, Flatten) == True):
      assert layer.layer_id != 0, "flatten can not be the 1st layer"
      prev_layer = self._layers[layer.layer_id-1]
      layer.calculate_inout_shape(prev_layer.output_shape)
    elif (isinstance(layer, Dense) == True):
      if (layer.layer_id > 0):
        prev_layer = self._layers[layer.layer_id-1]
        layer.calculate_inout_shape(prev_layer.output_shape[1])
        
    if (isinstance(layer, Conv2D) == True):
      layer.handle = self.ffmodel.conv2d_v2(layer.name, layer.in_channels, layer.out_channels, layer.kernel_size[0], layer.kernel_size[1], layer.stride[0], layer.stride[1], layer.padding[0], layer.padding[1], layer.activation, layer.use_bias)
    elif (isinstance(layer, MaxPooling2D) == True):
      layer.handle = self.ffmodel.pool2d_v2(layer.name, layer.kernel_size[1], layer.kernel_size[0], layer.stride[0], layer.stride[1], layer.padding[0], layer.padding[1])
    elif (isinstance(layer, Flatten) == True):
      layer.handle = self.ffmodel.flat_v2(layer.name)
    elif (isinstance(layer, Dense) == True):
      layer.handle = self.ffmodel.dense_v2(layer.name, layer.in_channels, layer.out_channels, layer.activation)
    elif (isinstance(layer, Activation) == True):
      print("add softmax")
    else:
      assert 0, "unknow layer"
      
  def compile(self, optimizer):
    self.ffoptimizer = optimizer
    optimizer.handle = ff.SGDOptimizer(self.ffmodel, optimizer.learning_rate)
    self.ffmodel.set_sgd_optimizer(self.ffoptimizer.handle)
    
  def fit(self, input_tensor, label_tensor, dataloader, alexnetconfig):
    if (self.use_v2 == True):
      self._init_inout(input_tensor, label_tensor)
    else:
      self._create_layer_and_init_inout(input_tensor, label_tensor)        
    self.ffmodel.init_layers()
    
    epochs = self.ffconfig.get_epochs()
  
    ts_start = self.ffconfig.get_current_time()
    for epoch in range(0,epochs):
      dataloader.reset()
      self.ffmodel.reset_metrics()
      iterations = dataloader.get_num_samples() / self.ffconfig.get_batch_size()
    
      for iter in range(0, int(iterations)):
        if (len(alexnetconfig.dataset_path) == 0):
          if (iter == 0 and epoch == 0):
            dataloader.next_batch(self.ffmodel)
        else:
          dataloader.next_batch(self.ffmodel)
        if (epoch > 0):
          ffconfig.begin_trace(111)
        self.ffmodel.forward()
        self.ffmodel.zero_gradients()
        self.ffmodel.backward()
        self.ffmodel.update()
        if (epoch > 0):
          self.ffconfig.end_trace(111)

    ts_end = self.ffconfig.get_current_time()
    run_time = 1e-6 * (ts_end - ts_start);
    print("epochs %d, ELAPSED TIME = %.4fs, THROUGHPUT = %.2f samples/s\n" %(epochs, run_time, dataloader.get_num_samples() * epochs / run_time));
    
    