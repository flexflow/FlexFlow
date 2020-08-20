# Copyright 2020 Stanford University, Los Alamos National Laboratory
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import flexflow.core as ff
from flexflow.core.flexflow_logger import fflogger

from .tensor import Tensor
from flexflow.keras.layers import Conv2D, Pooling2D, Flatten, Dense, Activation, Concatenate, Add, Subtract, Multiply, Dropout, BatchNormalization, Embedding
from flexflow.keras.optimizers import SGD, Adam 
from flexflow.keras.callbacks import Callback, LearningRateScheduler, VerifyMetrics, EpochVerifyMetrics 
from flexflow.keras import losses as keras_losses
from flexflow.keras import metrics as keras_metrics

from PIL import Image

tracing_id = 100

class BaseModel(object):
  __slots__ = ['_ffconfig', '_ffmodel', '_ffoptimizer', '_layers', '_nb_layers', \
               '_input_layers', '_input_tensors', '_output_tensor', '_label_tensor', \
               '_full_input_tensors', '_full_label_tensor', '_num_samples',\
               '_input_dataloaders', '_input_dataloaders_dim', \
               '_label_dataloader', '_label_dataloader_dim', \
               '_loss', '_metrics', '_label_type', '__tracing_id']
  def __init__(self, name):
    self._ffconfig = ff.FFConfig()
    self._ffconfig.parse_args()
    print("Python API batchSize(%d) workersPerNodes(%d) numNodes(%d)" %(self._ffconfig.get_batch_size(), self._ffconfig.get_workers_per_node(), self._ffconfig.get_num_nodes()))
    self._ffmodel = None
    
    self._name = name
    self._ffoptimizer = None
    self._layers = []
    self._nb_layers = 0
    self._input_layers = []
    self._input_tensors = []
    self._output_tensor = 0
    self._label_tensor = 0
    self._full_input_tensors = []
    self._full_label_tensor = 0
    self._num_samples = 0
    self._input_dataloaders = []
    self._input_dataloaders_dim = []
    self._label_dataloader = 0
    self._label_dataloader_dim = 0
    self._loss = None
    self._metrics = []
    self._label_type = ff.DataType.DT_FLOAT   
 
    global tracing_id
    self.__tracing_id = tracing_id
    tracing_id += 1
    
  @property
  def input(self):
    return self._input_tensors
  
  @property
  def output(self):
    return self._output_tensor
  
  @property  
  def layers(self):
    return self._layers
    
  @property
  def optimizer(self):
    return self._ffoptimizer
    
  @property
  def ffmodel(self):
    return self._ffmodel
    
  @property
  def ffconfig(self):
    return self._ffconfig
    
  def get_layer(self, name=None, index=None):
    if (index is not None):
      if (self._nb_layers <= index):
        raise ValueError('Was asked to retrieve layer at index ' +
                         str(index) + ' but model only has ' +
                         str(self._nb_layers) + ' layers.')
      else:
        return self._layers[index]
    else:
      if not name:
        raise ValueError('Provide either a layer name or layer index.')
    for layer in self._layers:
      if (layer.name == name):
        return layer
    raise ValueError('No such layer: ' + name)
  
  # TODO: finish API    
  def summary(self, line_length=None, positions=None, print_fn=None):
    if line_length != None:
      assert 0, "line_length is not supported"
    if print_fn != None:
      assert 0, "print_fn is not supported"
      
    model_summary = "Layer (type)\t\tOutput Shape\t\tInput Shape\tConnected to\n"
    for layer in self._input_layers:
      layer_summary = layer.get_summary()
      model_summary += layer_summary
    for layer in self._layers:
      fflogger.debug(str(layer))
      for prev_layer in layer.prev_layers:
        fflogger.debug("\tprev: %s" %( str(prev_layer)))
      for next_layer in layer.next_layers:
        fflogger.debug("\tnext: %s" %( str(next_layer)))
      layer_summary = layer.get_summary()
      model_summary += layer_summary 
      
    return model_summary
  
  #TODO: finish API  
  def compile(self,  
              optimizer, 
              loss=None, 
              metrics=None, 
              loss_weights=None, 
              weighted_metrics=None, 
              run_eagerly=None, 
              **kwargs):
    if loss_weights != None:
      assert 0, "loss_weights is not supported"
    if weighted_metrics != None:
      assert 0, "weighted_metrics is not supported"
    if run_eagerly != None:
      assert 0, "run_eagerly is not supported"
    
    assert loss != None, "loss is None"  
    if isinstance(loss, keras_losses.Loss) == True:
      self._loss = loss
    elif loss == 'categorical_crossentropy':
      self._loss = keras_losses.CategoricalCrossentropy()
    elif loss == 'sparse_categorical_crossentropy':
      self._loss = keras_losses.SparseCategoricalCrossentropy()
      self._label_type = ff.DataType.DT_INT32
    elif loss == 'mean_squared_error':
      self._loss = keras_losses.MeanSquaredError()
    else:
      assert 0, 'Unsupported loss'
    
    assert metrics != None, "metrics is None"
    assert isinstance(metrics, list) == True, 'Metrics should be a list'  
    for metric in metrics:
      if isinstance(metric, keras_metrics.Metric) == True:
        self._metrics.append(metric)
      elif metric == 'accuracy':
        self._metrics.append(keras_metrics.Accuracy())
      elif metric == 'categorical_crossentropy':
        self._metrics.append(keras_metrics.CategoricalCrossentropy())
      elif metric == 'sparse_categorical_crossentropy':
        self._metrics.append(keras_metrics.SparseCategoricalCrossentropy())
      elif metric == 'mean_squared_error':
        self._metrics.append(keras_metrics.MeanSquaredError())
      elif metric == 'root_mean_squared_error':
        self._metrics.append(keras_metrics.RootMeanSquaredError())
      elif metric == 'mean_absolute_error':
        self._metrics.append(keras_metrics.MeanAbsoluteError())
      else:
        assert 0, 'Unsupported metric'
    
    self._ffmodel = ff.FFModel(self._ffconfig)  
    self._create_input_tensors()
    self._create_flexflow_layers()
    
    self._verify_output_tensors()
    self._verify_input_tensors()
    
    self._ffoptimizer = optimizer
    self._create_optimizer()
    metrics_type = []
    for metric in self._metrics:
      metrics_type.append(metric.type)
    self._ffmodel.compile(optimizer=self._ffoptimizer.ffhandle, loss_type=self._loss.type, metrics=metrics_type)
    self._create_label_tensor()
    fflogger.debug("%s, %s, %s, %s" %( str(self._input_tensors[0]), str(self._output_tensor), str(self._input_tensors[0].ffhandle), str(self._output_tensor.ffhandle)))
  
  #TODO: finish API  
  def fit(self, 
          x=None, 
          y=None, 
          batch_size=None, 
          epochs=1, 
          verbose=1, 
          callbacks=None, 
          validation_split=0.0, 
          validation_data=None, 
          shuffle=True, 
          class_weight=None, 
          sample_weight=None, 
          initial_epoch=0, 
          steps_per_epoch=None, 
          validation_steps=None, 
          validation_batch_size=None, 
          validation_freq=1, 
          max_queue_size=10, 
          workers=1, 
          use_multiprocessing=False):
    if (batch_size != None):
      assert self._ffconfig.get_batch_size() == batch_size, "batch size is not correct use -b to set it"
    if validation_split != 0.0:
      assert 0, "validation_split is not supported"  
    if validation_data != None:
      assert 0, "validation_data is not supported"
    if shuffle != True:
      assert 0, "shuffle is not supported"
    if class_weight != None:
      assert 0, "class_weight is not supported"
    if sample_weight != None:
      assert 0, "sample_weight is not supported"
    if initial_epoch != 0:
      assert 0, "initial_epoch is not supported"
    if steps_per_epoch != None:
      assert 0, "steps_per_epoch is not supported"
    if validation_steps != None:
      assert 0, "validation_steps is not supported"
    if validation_batch_size != None:
      assert 0, "validation_batch_size is not supported"
    if validation_freq != 1:
      assert 0, "validation_freq is not supported"
    if max_queue_size != 10:
      assert 0, "max_queue_size is not supported"
    if workers != 1:
      assert 0, "workers is not supported"
    if use_multiprocessing != False:
      assert 0, "use_multiprocessing is not supported"
      
    assert self._output_tensor.ffhandle != None, "tensor is not init"
    if (isinstance(x, list) == False):
      input_tensors = [x]
    else:
      input_tensors = x
    label_tensor = y
    self._verify_tensors(input_tensors, label_tensor)
    self._create_data_loaders(input_tensors, label_tensor)
    self._ffmodel.init_layers()
    self._train(epochs, callbacks)
    
  def _create_input_tensor(self, idx):
    assert self._input_tensors[idx].batch_shape[0] != 0, "batch size is not set"
    self._input_tensors[idx].create_ff_tensor(self._ffmodel)
    
  def _create_label_tensor(self):
    label_ffhandle = self._ffmodel.get_label_tensor()
    self._label_tensor = Tensor(ffmodel=self._ffmodel, batch_shape=(self._ffconfig.get_batch_size(), 1), name="", dtype=self._label_type, ffhandle=label_ffhandle)
    
  def _create_input_tensors(self):
    idx = 0
    for input_tensor in self._input_tensors:
      input_tensor.set_batch_size(self._ffconfig.get_batch_size())
      self._create_input_tensor(idx)
      idx += 1
    
  def _verify_tensors(self, input_arrays, label_array):
    assert len(input_arrays) == len(self._input_tensors), "check len of input tensors"
    # TODO: move check shape into another function
    for np_array, t in zip(input_arrays, self._input_tensors):
      np_shape = np_array.shape
      assert len(np_shape) == t.num_dims, "check input shape"
      for i in range(1, len(np_shape)):
        assert np_shape[i] == t.batch_shape[i], "check input dims"
      assert np_array.dtype == t.dtype_str, "check input dtype"   

    np_shape = label_array.shape
    assert len(np_shape) == self._label_tensor.num_dims, "check label shape"
    for i in range(1, len(np_shape)):
      assert np_shape[i] == self._label_tensor.batch_shape[i], "check label dims"
    assert label_array.dtype == self._label_tensor.dtype_str
      
  def _verify_output_tensors(self):
    assert self._layers[self._nb_layers-1].output_tensors[0] == self._output_tensor, "output tensor is wrong"
    
  def _verify_input_tensors(self):
    for t in self._input_tensors:
      assert len(t.to_layers) > 0, "input tensor has not to_layers"
      
  def _create_optimizer(self):
    assert self._ffoptimizer != None, "optimizer is not set"
    if (isinstance(self._ffoptimizer, SGD) == True) or (isinstance(self._ffoptimizer, Adam) == True):
      self._ffoptimizer.create_ffhandle(self._ffmodel)
    else:
      assert 0, "unknown optimizer"
    
  def __create_single_data_loader(self, batch_tensor, full_array):
    array_shape = full_array.shape
    num_dim = len(array_shape)
    print("dataloader type:", full_array.dtype)
    if (full_array.dtype == "float32"):
      datatype = ff.DataType.DT_FLOAT
    elif (full_array.dtype == "int32"):
      datatype = ff.DataType.DT_INT32
    else:
      assert 0, "unsupported datatype"

    if (num_dim == 2):
      full_tensor = Tensor(self._ffmodel, batch_shape=[self._num_samples, array_shape[1]], name="", dtype=datatype)
    elif (num_dim == 4):
      full_tensor = Tensor(self._ffmodel, batch_shape=[self._num_samples, array_shape[1], array_shape[2], array_shape[3]], name="", dtype=datatype)
    else:
      assert 0, "unsupported dims"
      
    full_tensor.ffhandle.attach_numpy_array(self._ffconfig, full_array)
    dataloader = ff.SingleDataLoader(self._ffmodel, batch_tensor.ffhandle, full_tensor.ffhandle, self._num_samples, datatype) 
    full_tensor.ffhandle.detach_numpy_array(self._ffconfig)
    
    return full_tensor, dataloader
    
  def _create_data_loaders(self, x_trains, y_train):
    # Todo: check all num_samples, should be the same
    input_shape = x_trains[0].shape
    self._num_samples = input_shape[0]
    
    assert len(self._input_tensors) != 0, "input_tensor is not set"
    assert self._label_tensor != 0, "label_tensor is not set"
    
    idx = 0
    for x_train in x_trains:
      full_tensor, dataloader = self.__create_single_data_loader(self._input_tensors[idx], x_train)
      self._full_input_tensors.append(full_tensor)
      self._input_dataloaders.append(dataloader)
      self._input_dataloaders_dim.append(len(input_shape))
      idx += 1
    full_tensor, dataloader = self.__create_single_data_loader(self._label_tensor, y_train)
    self.__full_label_tensor = full_tensor
    self._label_dataloader = dataloader
    self._label_dataloader_dim = len(input_shape)
    
  def _train(self, epochs, callbacks):
    if callbacks != None:
      for callback in callbacks:
        callback.set_model(self)
        
    if callbacks != None:
      for callback in callbacks:
        callback.on_train_begin()
        
    ts_start = self._ffconfig.get_current_time()
    epoch = 0
    epoch_flag = True
    while (epoch < epochs) and (epoch_flag == True):
      if callbacks != None:
        for callback in callbacks:
          callback.on_epoch_begin(epoch)
      
      for dataloader in self._input_dataloaders:
        dataloader.reset()
      self._label_dataloader.reset()
      self._ffmodel.reset_metrics()
      iterations = self._num_samples / self._ffconfig.get_batch_size()

      for iter in range(0, int(iterations)):
        if callbacks != None:
          for callback in callbacks:
            callback.on_batch_begin(iter)
            
        for dataloader in self._input_dataloaders:
          dataloader.next_batch(self._ffmodel)
        self._label_dataloader.next_batch(self._ffmodel)
        if (epoch > 0):
          self._ffconfig.begin_trace(self.__tracing_id)
        self._ffmodel.forward()
        # for layer in self._layers:
        #   layer.ffhandle.forward(self._ffmodel)
        self._ffmodel.zero_gradients()
        self._ffmodel.backward()
        self._ffmodel.update()
        if (epoch > 0):
          self._ffconfig.end_trace(self.__tracing_id)
          
        if callbacks != None:
          for callback in callbacks:
            callback.on_batch_end(iter)
      
      if callbacks != None:
        for callback in callbacks:
          early_stop = callback.on_epoch_end(epoch)
          if early_stop == True:
            print("Accuracy reaches, now early stop, epoch: %d" %(epoch))
            epoch_flag = False
          
      epoch += 1 

    ts_end = self._ffconfig.get_current_time()
    run_time = 1e-6 * (ts_end - ts_start);
    print("epochs %d, ELAPSED TIME = %.4fs, interations %d, samples %d, THROUGHPUT = %.2f samples/s\n" %(epochs, run_time, int(iterations), self._num_samples, self._num_samples * epochs / run_time));
    
    if callbacks != None:
      for callback in callbacks:
        callback.on_train_end()

    # self._input_tensors[0].ffhandle.inline_map(self._ffconfig)
    # input_array = self._input_tensors[0].ffhandle.get_flat_array(self._ffconfig, ff.DataType.DT_FLOAT)
    # print(input_array.shape)
    # print(input_array)
    # #self.save_image(input_array, 2)
    # self._input_tensors[0].ffhandle.inline_unmap(self._ffconfig)
    #
    # self._label_tensor.ffhandle.inline_map(self._ffconfig)
    # label_array = self._label_tensor.ffhandle.get_flat_array(self._ffconfig, self._label_type)
    # print(label_array.shape)
    # print(label_array)
    # self._label_tensor.ffhandle.inline_unmap(self._ffconfig)
    
  def _create_flexflow_layers_v2(self):
    for layer in self._layers:

      if (isinstance(layer, Conv2D) == True):
        layer.ffhandle = self._ffmodel.conv2d_v2(layer.in_channels, layer.out_channels, layer.kernel_size[0], layer.kernel_size[1], layer.stride[0], layer.stride[1], layer.padding[0], layer.padding[1], layer.activation, layer.use_bias)
      elif (isinstance(layer, Pooling2D) == True):
        layer.ffhandle = self._ffmodel.pool2d_v2(layer.kernel_size[1], layer.kernel_size[0], layer.stride[0], layer.stride[1], layer.padding[0], layer.padding[1], layer.pool_type)
      elif (isinstance(layer, Flatten) == True):
        layer.ffhandle = self._ffmodel.flat_v2()
      elif (isinstance(layer, Dense) == True):
        layer.ffhandle = self._ffmodel.dense_v2(layer.in_channels, layer.out_channels, layer.activation, layer.use_bias)
      elif (isinstance(layer, Activation) == True):
        print("add softmax")
      elif (isinstance(layer, Concatenate) == True):
        print("add concatenate")
      else:
        assert 0, "unknow layer"
        
  def _create_flexflow_layers(self):
    out_t = 0
    for layer in self._layers:

      if isinstance(layer, Activation) == True:
        if layer.activation == 'softmax':
          assert layer.layer_id == self._nb_layers-1, "softmax is not in the last layer"
          out_t = self._ffmodel.softmax(layer.input_tensors[0].ffhandle)
        elif layer.activation == 'relu':
          out_t = self._ffmodel.relu(layer.input_tensors[0].ffhandle)
        elif layer.activation == 'sigmoid':
          out_t = self._ffmodel.sigmoid(layer.input_tensors[0].ffhandle)
        elif layer.activation == 'tanh':
          out_t = self._ffmodel.tanh(layer.input_tensors[0].ffhandle)
        elif layer.activation == 'elu':
          out_t = self._ffmodel.elu(layer.input_tensors[0].ffhandle)
      elif isinstance(layer, Concatenate) == True:
        t_ffhandle_list = []
        for t in layer.input_tensors:
          t_ffhandle_list.append(t.ffhandle)
        out_t = self._ffmodel.concat(t_ffhandle_list, layer.axis)
      elif isinstance(layer, Conv2D) == True:
        out_t = self._ffmodel.conv2d(layer.input_tensors[0].ffhandle, layer.out_channels, layer.kernel_size[0], layer.kernel_size[1], layer.stride[0], layer.stride[1], layer.padding[0], layer.padding[1], layer.activation, layer.use_bias, layer.kernel_initializer.ffhandle, layer.bias_initializer.ffhandle)
      elif isinstance(layer, Pooling2D) == True:
        out_t = self._ffmodel.pool2d(layer.input_tensors[0].ffhandle, layer.kernel_size[1], layer.kernel_size[0], layer.stride[0], layer.stride[1], layer.padding[0], layer.padding[1], layer.pool_type)
      elif isinstance(layer, Flatten) == True:
        out_t = self._ffmodel.flat(layer.input_tensors[0].ffhandle)
      elif isinstance(layer, Dense) == True:
        out_t = self._ffmodel.dense(layer.input_tensors[0].ffhandle, layer.out_channels, layer.activation, layer.use_bias, layer.kernel_initializer.ffhandle, layer.bias_initializer.ffhandle)
      elif isinstance(layer, Add) == True:
        out_t = self._ffmodel.add(layer.input_tensors[0].ffhandle, layer.input_tensors[1].ffhandle)
      elif isinstance(layer, Subtract) == True:
        out_t = self._ffmodel.subtract(layer.input_tensors[0].ffhandle, layer.input_tensors[1].ffhandle)
      elif isinstance(layer, Multiply) == True:
        out_t = self._ffmodel.multiply(layer.input_tensors[0].ffhandle, layer.input_tensors[1].ffhandle)
      elif isinstance(layer, Dropout) == True:
        out_t = self._ffmodel.dropout(layer.input_tensors[0].ffhandle, layer.rate, layer.seed)
      elif isinstance(layer, BatchNormalization) == True:
        out_t = self._ffmodel.batch_norm(layer.input_tensors[0].ffhandle)
      elif isinstance(layer, Embedding) == True:
        out_t = self._ffmodel.embedding(layer.input_tensors[0].ffhandle, layer.input_dim, layer.out_channels, ff.AggrMode.AGGR_MODE_SUM, layer.embeddings_initializer.ffhandle)
      else:
        assert 0, "unknow layer"

      layer.output_tensors[0].ffhandle = out_t
      layer.set_batch_size(self._ffconfig.get_batch_size())

      assert layer.ffhandle == None, "layer handle is inited"
      layer.ffhandle = self._ffmodel.get_layer_by_id(layer.layer_id)
      assert layer.ffhandle != None, "layer handle is wrong"
       
  def _init_inout(self):
    out_t = 0
    for layer in self._layers:

      if (isinstance(layer, Activation) == True):
        assert layer_id == self._nb_layers-1, "softmax is not in the last layer"
        out_t = self._ffmodel.softmax(layer.input_tensors[0].ffhandle)
        assert layer.ffhandle == 0, "layer handle is inited"
        layer.ffhandle = self._ffmodel.get_layer_by_id(layer_id)
      elif (isinstance(layer, Concatenate) == True):
        t_ffhandle_list = []
        for t in layer.input_tensors:
          t_ffhandle_list.append(t.ffhandle)
        out_t = self._ffmodel.concat(t_ffhandle_list, layer.axis)
        assert layer.ffhandle == 0, "layer handle is inited"
        layer.ffhandle = self._ffmodel.get_layer_by_id(layer_id)
      else:
        out_t = layer.ffhandle.init_inout(self._ffmodel, layer.input_tensors[0].ffhandle);
      
      layer.output_tensors[0].ffhandle = out_t
      layer.set_batch_size(self._ffconfig.get_batch_size())
      assert layer.ffhandle != None, "layer handle is wrong"
      
    print("output tensor", self._output_tensor.batch_shape)
    
  def save_image(self, batch_image_array, id):
    image_array = batch_image_array[id, :, :, :]
    image_array = image_array.transpose(1, 2, 0)
    image_array = image_array*255
    image_array = image_array.astype('uint8')
    pil_image = Image.fromarray(image_array).convert('RGB')
    pil_image.save("img.jpeg")
