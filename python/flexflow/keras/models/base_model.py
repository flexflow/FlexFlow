import flexflow.core as ff

from .input_layer import Tensor
from flexflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Concatenate
from flexflow.keras.optimizers import SGD, Adam 

from PIL import Image

class BaseModel(object):
  def __init__(self):
    self.ffconfig = ff.FFConfig()
    self.ffconfig.parse_args()
    print("Python API batchSize(%d) workersPerNodes(%d) numNodes(%d)" %(self.ffconfig.get_batch_size(), self.ffconfig.get_workers_per_node(), self.ffconfig.get_num_nodes()))
    self.ffmodel = ff.FFModel(self.ffconfig)
    
    self.ffoptimizer = 0
    self._layers = dict()
    self._nb_layers = 0
    self.input_tensors = []
    self.output_tensor = 0
    self.label_tensor = 0
    self.full_input_tensors = []
    self.full_label_tensor = 0
    self.num_samples = 0
    self.input_dataloaders = []
    self.input_dataloaders_dim = []
    self.label_dataloader = 0
    self.label_dataloader_dim = 0
    
  @property
  def input(self):
    return self.input_tensors
  
  @property
  def output(self):
    return self.output_tensor
    
  def get_layer(self, layer_id):
    return self._layers[layer_id]
    
  def _create_input_tensor(self, idx):
    assert self.input_tensors[idx].batch_shape[0] != 0, "batch size is not set"
    self.input_tensors[idx].create_ff_tensor(self.ffmodel)
    
  def _create_label_tensor(self):
    self.label_tensor = Tensor(self.ffmodel, batch_shape=[self.ffconfig.get_batch_size(), 1], name="", dtype="int32")
    
  def _create_input_and_label_tensors(self):
    idx = 0
    for input_tensor in self.input_tensors:
      input_tensor.set_batch_size(self.ffconfig.get_batch_size())
      self._create_input_tensor(idx)
      idx += 1

    self._create_label_tensor()
    
  def _verify_tensors(self, input_arrays, label_array):
    assert len(input_arrays) == len(self.input_tensors), "check len of input tensors"
    # TODO: move check shape into another function
    for np_array, t in zip(input_arrays, self.input_tensors):
      np_shape = np_array.shape
      assert len(np_shape) == t.num_dims, "check input shape"
      for i in range(1, len(np_shape)):
        assert np_shape[i] == t.batch_shape[i], "check input dims"
    np_shape = label_array.shape
    assert len(np_shape) == self.label_tensor.num_dims, "check label shape"
    for i in range(1, len(np_shape)):
      assert np_shape[i] == self.label_tensor.batch_shape[i], "check label dims"    
      
  def _verify_output_tensors(self):
    assert self._layers[self._nb_layers-1].output_tensor == self.output_tensor, "output tensor is wrong"
    
  def _verify_input_tensors(self):
    for t in self.input_tensors:
      assert len(t.to_layers) > 0, "input tensor has not to_layers"
    
  def _compile(self, optimizer):
    self.ffoptimizer = optimizer
      
  def _set_optimizer(self):
    assert self.ffoptimizer != 0, "optimizer is not set"
    if (isinstance(self.ffoptimizer, SGD) == True):
      self.ffoptimizer.ffhandle = ff.SGDOptimizer(self.ffmodel, self.ffoptimizer.learning_rate)
      self.ffmodel.set_sgd_optimizer(self.ffoptimizer.ffhandle)
    elif (isinstance(self.ffoptimizer, Adam) == True):
      self.ffoptimizer.ffhandle = ff.AdamOptimizer(self.ffmodel, self.ffoptimizer.learning_rate, self.ffoptimizer.beta1, self.ffoptimizer.beta2)
      self.ffmodel.set_adam_optimizer(self.ffoptimizer.ffhandle)
    else:
      assert 0, "unknown optimizer"
    
  def __create_single_data_loader(self, batch_tensor, full_array):
    array_shape = full_array.shape
    num_dim = len(array_shape)
    print(array_shape)
    
    if (full_array.dtype == "float32"):
      datatype = ff.DataType.DT_FLOAT
    elif (full_array.dtype == "int32"):
      datatype = ff.DataType.DT_INT32
    else:
      assert 0, "unsupported datatype"

    if (num_dim == 2):
      full_tensor = Tensor(self.ffmodel, batch_shape=[self.num_samples, array_shape[1]], name="", dtype=datatype)
    elif (num_dim == 4):
      full_tensor = Tensor(self.ffmodel, batch_shape=[self.num_samples, array_shape[1], array_shape[2], array_shape[3]], name="", dtype=datatype)
    else:
      assert 0, "unsupported dims"
      
    full_tensor.ffhandle.attach_numpy_array(self.ffconfig, full_array)
    dataloader = ff.SingleDataLoader(self.ffmodel, batch_tensor.ffhandle, full_tensor.ffhandle, self.num_samples, datatype) 
    full_tensor.ffhandle.detach_numpy_array(self.ffconfig)
    
    return full_tensor, dataloader
    
  def _create_data_loaders(self, x_trains, y_train):
    # Todo: check all num_samples, should be the same
    input_shape = x_trains[0].shape
    self.num_samples = input_shape[0]
    
    assert len(self.input_tensors) != 0, "input_tensor is not set"
    assert self.label_tensor != 0, "label_tensor is not set"
    
    print(y_train.shape)
    idx = 0
    for x_train in x_trains:
      full_tensor, dataloader = self.__create_single_data_loader(self.input_tensors[idx], x_train)
      self.full_input_tensors.append(full_tensor)
      self.input_dataloaders.append(dataloader)
      self.input_dataloaders_dim.append(len(input_shape))
      idx += 1
    full_tensor, dataloader = self.__create_single_data_loader(self.label_tensor, y_train)
    self.full_label_tensor = full_tensor
    self.label_dataloader = dataloader
    self.label_dataloader_dim = len(input_shape)
    
  def _train(self, epochs):
    ts_start = self.ffconfig.get_current_time()
    for epoch in range(0,epochs):
      for dataloader in self.input_dataloaders:
        dataloader.reset()
      self.label_dataloader.reset()
      self.ffmodel.reset_metrics()
      iterations = self.num_samples / self.ffconfig.get_batch_size()

      for iter in range(0, int(iterations)):
        for dataloader in self.input_dataloaders:
          dataloader.next_batch(self.ffmodel)
        self.label_dataloader.next_batch(self.ffmodel)
        if (epoch > 0):
          self.ffconfig.begin_trace(111)
        self.ffmodel.forward()
        # for layer_id in self._layers:
        #   layer = self._layers[layer_id]
        #   layer.ffhandle.forward(self.ffmodel)
        self.ffmodel.zero_gradients()
        self.ffmodel.backward()
        self.ffmodel.update()
        if (epoch > 0):
          self.ffconfig.end_trace(111)

    ts_end = self.ffconfig.get_current_time()
    run_time = 1e-6 * (ts_end - ts_start);
    print("epochs %d, ELAPSED TIME = %.4fs, interations %d, samples %d, THROUGHPUT = %.2f samples/s\n" %(epochs, run_time, int(iterations), self.num_samples, self.num_samples * epochs / run_time));

    self.input_tensors[0].ffhandle.inline_map(self.ffconfig)
    input_array = self.input_tensors[0].ffhandle.get_flat_array(self.ffconfig, ff.DataType.DT_FLOAT)
    print(input_array.shape)
    print(input_array)
    #self.save_image(input_array, 2)
    self.input_tensors[0].ffhandle.inline_unmap(self.ffconfig)
    
    self.label_tensor.ffhandle.inline_map(self.ffconfig)
    label_array = self.label_tensor.ffhandle.get_flat_array(self.ffconfig, ff.DataType.DT_INT32)
    print(label_array.shape)
    print(label_array)
    self.label_tensor.ffhandle.inline_unmap(self.ffconfig)
    
  def _create_flexflow_layers_v2(self):
    for layer_id in self._layers:
      layer = self._layers[layer_id]

      if (isinstance(layer, Conv2D) == True):
        layer.ffhandle = self.ffmodel.conv2d_v2(layer.name, layer.in_channels, layer.out_channels, layer.kernel_size[0], layer.kernel_size[1], layer.stride[0], layer.stride[1], layer.padding[0], layer.padding[1], layer.activation, layer.use_bias)
      elif (isinstance(layer, MaxPooling2D) == True):
        layer.ffhandle = self.ffmodel.pool2d_v2(layer.name, layer.kernel_size[1], layer.kernel_size[0], layer.stride[0], layer.stride[1], layer.padding[0], layer.padding[1])
      elif (isinstance(layer, Flatten) == True):
        layer.ffhandle = self.ffmodel.flat_v2(layer.name)
      elif (isinstance(layer, Dense) == True):
        layer.ffhandle = self.ffmodel.dense_v2(layer.name, layer.in_channels, layer.out_channels, layer.activation)
      elif (isinstance(layer, Activation) == True):
        print("add softmax")
      elif (isinstance(layer, Concatenate) == True):
        print("add concatenate")
      else:
        assert 0, "unknow layer"
        
  def _create_flexflow_layers(self, verify_inout_shape=True):
    out_t = 0
    for layer_id in self._layers:
      layer = self._layers[layer_id]

      if (isinstance(layer, Activation) == True):
       assert layer.layer_id == self._nb_layers-1, "softmax is not in the last layer"
       out_t = self.ffmodel.softmax("softmax", layer.input_tensors[0].ffhandle, self.label_tensor.ffhandle)
      elif (isinstance(layer, Concatenate) == True):
       t_ffhandle_list = []
       for t in layer.input_tensors:
         t_ffhandle_list.append(t.ffhandle)
       out_t = self.ffmodel.concat("concat", t_ffhandle_list, layer.axis)
      elif (isinstance(layer, Conv2D) == True):
       out_t = self.ffmodel.conv2d(layer.name, layer.input_tensors[0].ffhandle, layer.out_channels, layer.kernel_size[0], layer.kernel_size[1], layer.stride[0], layer.stride[1], layer.padding[0], layer.padding[1], layer.activation, layer.use_bias)
      elif (isinstance(layer, MaxPooling2D) == True):
       out_t = self.ffmodel.pool2d(layer.name, layer.input_tensors[0].ffhandle, layer.kernel_size[1], layer.kernel_size[0], layer.stride[0], layer.stride[1], layer.padding[0], layer.padding[1])
      elif (isinstance(layer, Flatten) == True):
       out_t = self.ffmodel.flat(layer.name, layer.input_tensors[0].ffhandle)
      elif (isinstance(layer, Dense) == True):
       out_t = self.ffmodel.dense(layer.name, layer.input_tensors[0].ffhandle, layer.out_channels, layer.activation)
      else:
       assert 0, "unknow layer"

      layer.output_tensor.set_ffhandle(out_t)

      assert layer.ffhandle == 0, "layer handle is inited"
      layer.ffhandle = self.ffmodel.get_layer_by_id(layer.layer_id)
      assert layer.ffhandle != 0, "layer handle is wrong"
      print(layer.ffhandle)    

      if (verify_inout_shape == True):
       in_t = layer.input_tensors[0].ffhandle
       layer.verify_inout_shape(in_t, out_t)
       
  def _init_inout(self, verify_inout_shape=True):
    out_t = 0
    for layer_id in self._layers:
      layer = self._layers[layer_id]

      if (isinstance(layer, Activation) == True):
        assert layer_id == self._nb_layers-1, "softmax is not in the last layer"
        out_t = self.ffmodel.softmax("softmax", layer.input_tensors[0].ffhandle, self.label_tensor.ffhandle)
        assert layer.ffhandle == 0, "layer handle is inited"
        layer.ffhandle = self.ffmodel.get_layer_by_id(layer_id)
      elif (isinstance(layer, Concatenate) == True):
        t_ffhandle_list = []
        for t in layer.input_tensors:
          t_ffhandle_list.append(t.ffhandle)
        out_t = self.ffmodel.concat("concat", t_ffhandle_list, layer.axis)
        assert layer.ffhandle == 0, "layer handle is inited"
        layer.ffhandle = self.ffmodel.get_layer_by_id(layer_id)
      else:
        out_t = layer.ffhandle.init_inout(self.ffmodel, layer.input_tensors[0].ffhandle);
      
      layer.output_tensor.set_ffhandle(out_t)
      assert layer.ffhandle != 0, "layer handle is wrong"
      print(layer.ffhandle)    
      
      if (verify_inout_shape == True):
        in_t = layer.input_tensors[0].ffhandle
        layer.verify_inout_shape(in_t, out_t)
    print("output tensor", self.output_tensor.batch_shape)
    
  def summary(self):
    model_summary = "Layer (type)\t\tOutput Shape\t\tInput Shape\tConnected to\n"
    for layer_id in self._layers:
      layer = self._layers[layer_id]
      print(layer)
      for prev_layer in layer.prev_layers:
        print("\tprev:  ", prev_layer)
      for next_layer in layer.next_layers:
        print("\tnext:  ", next_layer)
      layer_summary = layer.get_summary()
      model_summary += layer_summary 
      
    return model_summary
    
  def save_image(self, batch_image_array, id):
    image_array = batch_image_array[id, :, :, :]
    image_array = image_array.transpose(1, 2, 0)
    image_array = image_array*255
    image_array = image_array.astype('uint8')
    pil_image = Image.fromarray(image_array).convert('RGB')
    pil_image.save("img.jpeg")