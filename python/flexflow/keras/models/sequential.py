import flexflow.core as ff

from .base_model import BaseModel
from .input_layer import Tensor
from flexflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation

class Sequential(BaseModel):
  def __init__(self):
    super(Sequential, self).__init__() 

    self.use_v2 = False
    
  def _create_layer_and_init_inout(self, verify_inout_shape=True):
    int_t = 0
    out_t = 0
    for layer_id in self._layers:
      layer = self._layers[layer_id]
      assert layer.layer_id == layer_id, "wrong layer id"
      if (layer_id == 0):
        in_t = self.input_tensor.ffhandle
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
        out_t = self.ffmodel.softmax("softmax", in_t, self.label_tensor.ffhandle)

      if (verify_inout_shape == True):
        layer.verify_inout_shape(in_t, out_t)
        
      layer.ffhandle = self.ffmodel.get_layer_by_id(layer_id)
      print(layer.ffhandle)
    self.output_tensor = Tensor(dtype=self.input_tensor.dtype, ffhandle=out_t)
    print("output tensor", self.output_tensor.batch_shape)
    
  def _init_inout(self, verify_inout_shape=True):
    int_t = 0
    out_t = 0
    for layer_id in self._layers:
      layer = self._layers[layer_id]
      if (layer_id == 0):
        in_t = self.input_tensor.ffhandle
        out_t = layer.ffhandle.init_inout(self.ffmodel, in_t);
      else:
        in_t = out_t
        if (isinstance(layer, Activation) == True):
          assert layer_id == self._nb_layers-1, "softmax is not in the last layer"
          out_t = self.ffmodel.softmax("softmax", in_t, self.label_tensor.ffhandle)
          assert layer.ffhandle == 0, "layer handle is inited"
          layer.ffhandle = self.ffmodel.get_layer_by_id(layer_id)
        else:
          out_t = layer.ffhandle.init_inout(self.ffmodel, in_t);
      
      assert layer.ffhandle != 0, "layer handle is wrong"
      print(layer.ffhandle)    
      
      if (verify_inout_shape == True):
        layer.verify_inout_shape(in_t, out_t)
    self.output_tensor = Tensor(dtype=self.input_tensor.dtype, ffhandle=out_t)
    print("output tensor", self.output_tensor.batch_shape)
    
  def add_v1(self, layer):
    self._layers[self._nb_layers] = layer
    assert layer.layer_id == -1, "layer id is inited"
    assert layer.ffhandle == 0, "layer handle is inited"
    layer.layer_id = self._nb_layers
    self._nb_layers += 1
    
    prev_layer = 0

    if (isinstance(layer, Conv2D) == True):
      if (layer.layer_id > 0):
        prev_layer = self._layers[layer.layer_id-1]
        assert len(prev_layer.output_shape) == 4, "check prev layer"
        layer.calculate_inout_shape(prev_layer.output_shape[1], prev_layer.output_shape[2], prev_layer.output_shape[3], prev_layer.output_shape[0])
    elif (isinstance(layer, MaxPooling2D) == True):
      assert layer.layer_id != 0, "maxpool2d can not be the 1st layer"
      prev_layer = self._layers[layer.layer_id-1]
      assert len(prev_layer.output_shape) == 4, "check prev layer"
      layer.calculate_inout_shape(prev_layer.output_shape[1], prev_layer.output_shape[2], prev_layer.output_shape[3], prev_layer.output_shape[0])
    elif (isinstance(layer, Flatten) == True):
      assert layer.layer_id != 0, "flatten can not be the 1st layer"
      prev_layer = self._layers[layer.layer_id-1]
      layer.calculate_inout_shape(prev_layer.output_shape)
    elif (isinstance(layer, Dense) == True):
      if (layer.layer_id > 0):
        prev_layer = self._layers[layer.layer_id-1]
        assert len(prev_layer.output_shape) == 2, "check prev layer"
        layer.calculate_inout_shape(prev_layer.output_shape[1], prev_layer.output_shape[0])
    else:
      prev_layer = self._layers[layer.layer_id-1]
    
    layer.verify_meta_data()
    
    if (prev_layer != 0):
      layer.add_prev_layer(prev_layer)
      prev_layer.add_next_layer(layer)
  
  def add(self, layer):
    self.use_v2 = True
    self._layers[self._nb_layers] = layer
    assert layer.layer_id == -1, "layer id is inited"
    assert layer.ffhandle == 0, "layer handle is inited"
    layer.layer_id = self._nb_layers
    self._nb_layers += 1
    
    prev_layer = 0

    if (isinstance(layer, Conv2D) == True):
      if (layer.layer_id > 0):
        prev_layer = self._layers[layer.layer_id-1]
        assert len(prev_layer.output_shape) == 4, "check prev layer"
        layer.calculate_inout_shape(prev_layer.output_shape[1], prev_layer.output_shape[2], prev_layer.output_shape[3], prev_layer.output_shape[0])
    elif (isinstance(layer, MaxPooling2D) == True):
      assert layer.layer_id != 0, "maxpool2d can not be the 1st layer"
      prev_layer = self._layers[layer.layer_id-1]
      assert len(prev_layer.output_shape) == 4, "check prev layer"
      layer.calculate_inout_shape(prev_layer.output_shape[1], prev_layer.output_shape[2], prev_layer.output_shape[3], prev_layer.output_shape[0])
    elif (isinstance(layer, Flatten) == True):
      assert layer.layer_id != 0, "flatten can not be the 1st layer"
      prev_layer = self._layers[layer.layer_id-1]
      layer.calculate_inout_shape(prev_layer.output_shape)
    elif (isinstance(layer, Dense) == True):
      if (layer.layer_id > 0):
        prev_layer = self._layers[layer.layer_id-1]
        assert len(prev_layer.output_shape) == 2, "check prev layer"
        layer.calculate_inout_shape(prev_layer.output_shape[1], prev_layer.output_shape[0])
    else:
      prev_layer = self._layers[layer.layer_id-1]
        
    layer.verify_meta_data()
    
    if (prev_layer != 0):
      layer.add_prev_layer(prev_layer)
      prev_layer.add_next_layer(layer)

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
    else:
      assert 0, "unknow layer"
    
  def create_input_and_label_tensor(self, input_shape, label_shape):
    if (len(input_shape) == 2):
      self.input_tensor = Tensor(self.ffmodel, batch_shape=[self.ffconfig.get_batch_size(), input_shape[1]], name="", dtype="float32")
      
    elif (len(input_shape) == 4):
      self.input_tensor = Tensor(self.ffmodel, batch_shape=[self.ffconfig.get_batch_size(), input_shape[1], input_shape[2], input_shape[3]], name="", dtype="float32")
    
    self.label_tensor = Tensor(self.ffmodel, batch_shape=[self.ffconfig.get_batch_size(), 1], name="", dtype="int32")
    
  def fit(self, input_tensor, label_tensor, epochs=1):
    self.create_input_and_label_tensor(input_tensor.shape, label_tensor.shape)
    self._create_data_loaders(input_tensor, label_tensor)
    
    # if (self.use_v2 == True):
    self._init_inout()
    # else:
    #self._create_layer_and_init_inout()   
    self._set_optimizer()     
    self.ffmodel.init_layers()
    
    self._train(epochs)
    
  def summary(self):
    model_summary = "Layer (type)\t\tOutput Shape\t\tInput Shape\n"
    for layer_id in self._layers:
      layer = self._layers[layer_id]
      layer_summary = layer.get_summary()
      model_summary += layer_summary
      
    # layer = self._layers[0]
    # layer_summary = layer.get_summary()
    # model_summary += layer_summary
    # while (len(layer.next_layers) != 0):
    #    layer = layer.next_layers[0]
    #    layer_summary = layer.get_summary()
    #    model_summary += layer_summary
    
    # test_layer = Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding=(1,1), activation="relu")
    # test_layer.add_next_layer(self._layers[2])
    # self._layers[0].add_next_layer(test_layer)
    bfs_queue = []
    layer = self._layers[0]
    bfs_queue.append(layer)
    while(len(bfs_queue) != 0):
      layer = bfs_queue.pop(0)
      print(layer)
      for child in layer.next_layers:
        if child not in bfs_queue:
          bfs_queue.append(child)
        else:
          print(child, "already in the queue")
    
      
    return model_summary
    
    