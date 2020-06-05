import flexflow.core as ff

from .base_model import BaseModel
from .input_layer import Tensor
from flexflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Concatenate

class Model(BaseModel):
  def __init__(self, input_tensors, output_tensor):
    super(Model, self).__init__()
    
    if (isinstance(input_tensors, list) == False):
       input_tensors = [input_tensors]
       
    self.input_tensors = input_tensors
    self.output_tensor = output_tensor
    
    idx = 0
    for input_tensor in self.input_tensors:
      input_tensor.set_batch_size(self.ffconfig.get_batch_size())
      self.create_input_tensor(input_tensor.batch_shape, idx)
      idx += 1

    label_shape = (self.ffconfig.get_batch_size(), 1)
    self.create_label_tensor(label_shape)
    
    bfs_queue = []
    
    use_api = 2
    
    if (use_api == 1):  
      # for input_tensor in self.input_tensors:
      #   for layer in input_tensor.to_layers:
      #     bfs_queue.append(layer)
      # while(len(bfs_queue) != 0):
      #   layer = bfs_queue.pop(0)
      #   #print(layer)
      #   self._add_layer(layer)
      #   for child in layer.next_layers:
      #     if child not in bfs_queue:
      #       bfs_queue.append(child)
      #     else:
      #       print(child, "already in the queue")
      
      for input_tensor in reversed(self.input_tensors):
        for layer in reversed(input_tensor.to_layers):
          bfs_queue.append(layer)
      while(len(bfs_queue) != 0):
        layer = bfs_queue.pop()
        #print(layer)
        self._add_layer(layer)
        for child in reversed(layer.next_layers):
          assert child not in bfs_queue, "already in the stack"
          if child.nb_visited_prev_layers == len(child.prev_layers)-1:
            bfs_queue.append(child)
          else:
            child.nb_visited_prev_layers += 1
      
      self._init_inout()
    else:
      # for input_tensor in self.input_tensors:
      #   for layer in input_tensor.to_layers:
      #     bfs_queue.append(layer)
      # while(len(bfs_queue) != 0):
      #   layer = bfs_queue.pop(0)
      #   #print(layer)
      #   self._add_layer_and_init_inout(layer)
      #   for child in layer.next_layers:
      #     if child not in bfs_queue:
      #       bfs_queue.append(child)
      #     else:
      #       print(child, "already in the queue")
      
      for input_tensor in reversed(self.input_tensors):
        for layer in reversed(input_tensor.to_layers):
          bfs_queue.append(layer)
      while(len(bfs_queue) != 0):
        layer = bfs_queue.pop()
        #print(layer)
        self._add_layer_and_init_inout(layer)
        #self._add_layer_metadata(layer)
        for child in reversed(layer.next_layers):
          assert child not in bfs_queue, "already in the stack"
          if child.nb_visited_prev_layers == len(child.prev_layers)-1:
            bfs_queue.append(child)
          else:
            child.nb_visited_prev_layers += 1
      
      #self._init_layer_and_init_inout()
    
  def create_input_tensor(self, input_shape, idx):
    self.input_tensors[idx].create_ff_tensor(self.ffmodel)
    
  def create_label_tensor(self, label_shape): 
    self.label_tensor = Tensor(self.ffmodel, batch_shape=[self.ffconfig.get_batch_size(), 1], name="", dtype="int32")
  
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
    
  def _add_layer_metadata(self, layer):
    self._layers[self._nb_layers] = layer
    assert layer.layer_id == -1, "layer id is inited"
    assert layer.ffhandle == 0, "layer handle is inited"
    layer.layer_id = self._nb_layers
    self._nb_layers += 1
    
  def _init_layer_and_init_inout(self, verify_inout_shape=True):
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
    
  def _add_layer_and_init_inout(self, layer, verify_inout_shape=True):
    out_t = 0
    self._layers[self._nb_layers] = layer
    assert layer.layer_id == -1, "layer id is inited"
    assert layer.ffhandle == 0, "layer handle is inited"
    layer.layer_id = self._nb_layers
    self._nb_layers += 1

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
          
  def _add_layer(self, layer):
    self._layers[self._nb_layers] = layer
    assert layer.layer_id == -1, "layer id is inited"
    assert layer.ffhandle == 0, "layer handle is inited"
    layer.layer_id = self._nb_layers
    self._nb_layers += 1

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
      
  def compile(self, optimizer):
    self._compile(optimizer)
    
  def fit(self, input_tensors, label_tensor, epochs=1):
    assert self.output_tensor.ffhandle != 0, "tensor is not init"
    if (isinstance(input_tensors, list) == False):
       input_tensors = [input_tensors]
    self._verify_tensors(input_tensors, label_tensor)
    self._create_data_loaders(input_tensors, label_tensor)
    self._set_optimizer()     
    self.ffmodel.init_layers()
    self._train(epochs)
    