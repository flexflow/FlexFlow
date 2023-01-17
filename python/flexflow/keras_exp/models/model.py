# Copyright 2023 CMU, Facebook, LANL, MIT, NVIDIA, and Stanford (alphabetical)
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

from tensorflow.keras.models import Model as tf_keras_Model
from tensorflow.keras import optimizers as tf_keras_optimizer
#from tensorflow.keras import losses as tf_keras_losses
#from tensorflow.keras import metrics as tf_keras_metrics
import keras2onnx
import onnx

import flexflow.core as ff
from flexflow.core.flexflow_logger import fflogger

from .tensor import Tensor
from flexflow.keras import optimizers as ff_keras_optimizer
from flexflow.keras.callbacks import Callback, LearningRateScheduler, VerifyMetrics, EpochVerifyMetrics
from flexflow.keras import losses as ff_keras_losses
from flexflow.keras import metrics as ff_keras_metrics

from flexflow.onnx.model import ONNXModelKeras

tracing_id = 100

class BaseModel(object):
  def __init__(self, inputs, onnx_model):
    self._ffconfig = ff.FFConfig()
    print("Python API batchSize(%d) workersPerNodes(%d) numNodes(%d)" %(self._ffconfig.batch_size, self._ffconfig.workers_per_node, self._ffconfig.num_nodes))
    self._ffmodel = None
    self._onnx_model = onnx_model
    
    for node in onnx_model.graph.node:
      print(node)
      
    for input in onnx_model.graph.initializer:
      print(input.name, input.dims, len(input.dims))
    
    # for input in onnx_model.graph.input:
    #   print(input)
      
    self._input_tensors = []
    for key in inputs:
      input_tensor = inputs[key]
      t = Tensor(ffconfig=self._ffconfig, key=key, shape=input_tensor.shape, dtype=input_tensor.dtype)
      self._input_tensors.append(t)
    
    self._loss = None
    self._label_type = None
    self._metrics = []
    self._label_type = ff.DataType.DT_FLOAT
    self._my_onnx_model = None
    self._output_tensor = None
    self._num_samples = 0
    self._input_dataloaders = []
    self._input_dataloaders_dim = []
    self._label_dataloader = 0
    self._label_dataloader_dim = 0
    
    global tracing_id
    self.__tracing_id = tracing_id
    tracing_id += 1
    
  def compile(self,
              optimizer,
              loss=None,
              metrics=None,
              loss_weights=None,
              weighted_metrics=None,
              run_eagerly=None,
              comp_mode=None,
              **kwargs):
    if loss_weights != None:
      assert 0, "loss_weights is not supported"
    if weighted_metrics != None:
      assert 0, "weighted_metrics is not supported"
    if run_eagerly != None:
      assert 0, "run_eagerly is not supported"

    assert loss != None, "loss is None"
    if loss == 'categorical_crossentropy':
      self._loss = ff_keras_losses.CategoricalCrossentropy()
    elif loss == 'sparse_categorical_crossentropy':
      self._loss = ff_keras_losses.SparseCategoricalCrossentropy()
      self._label_type = ff.DataType.DT_INT32
    elif loss == 'mean_squared_error':
      self._loss = ff_keras_losses.MeanSquaredError()
    else:
      assert 0, 'Unsupported loss'

    assert metrics != None, "metrics is None"
    assert isinstance(metrics, list) == True, 'Metrics should be a list'
    for metric in metrics:
      if metric == 'accuracy':
        self._metrics.append(ff_keras_metrics.Accuracy())
      elif metric == 'categorical_crossentropy':
        self._metrics.append(ff_keras_metrics.CategoricalCrossentropy())
      elif metric == 'sparse_categorical_crossentropy':
        self._metrics.append(ff_keras_metrics.SparseCategoricalCrossentropy())
      elif metric == 'mean_squared_error':
        self._metrics.append(ff_keras_metrics.MeanSquaredError())
      elif metric == 'root_mean_squared_error':
        self._metrics.append(ff_keras_metrics.RootMeanSquaredError())
      elif metric == 'mean_absolute_error':
        self._metrics.append(ff_keras_metrics.MeanAbsoluteError())
      else:
        assert 0, 'Unsupported metric'
        
    self._ffmodel = ff.FFModel(self._ffconfig)
    self._create_input_tensors()
    self._create_flexflow_layers()
    
    layers = self._ffmodel.get_layers()
    for l in layers:
      print(l, layers[l])
    
    if isinstance(optimizer, tf_keras_optimizer.Optimizer) == True:
      if isinstance(optimizer, tf_keras_optimizer.SGD) == True:
        self._ffoptimizer = ff_keras_optimizer.SGD(learning_rate=optimizer.learning_rate.numpy(), momentum=optimizer.momentum.numpy(), nesterov=optimizer.nesterov)
      elif isinstance(optimizer, tf_keras_optimizer.Adam) == True:
        self._ffoptimizer = ff_keras_optimizer.Adam(learning_rate=optimizer.learning_rate.numpy(), beta_1=optimizer.beta_1.numpy(), beta_2=optimizer.beta_2.numpy(), epsilon=optimizer.epsilon.numpy())
      else:
        assert 0, "Unsupported optimizer"
    elif type(optimizer) == str:
      if optimizer == 'SGD':
        self._ffoptimizer = ff_keras_optimizer.SGD()
      elif optimizer == 'Adam':
        self._ffoptimizer = ff_keras_optimizer.Adam()
      else:
        assert 0, "Unsupported optimizer"
    else:
      assert 0, "Unsupported optimizer"

    self._create_optimizer()
    metrics_type = []
    for metric in self._metrics:
      metrics_type.append(metric.type)
    self._ffmodel.compile(optimizer=self._ffoptimizer.ffhandle, loss_type=self._loss.type, metrics=metrics_type, comp_mode=comp_mode)
    self._create_label_tensor()
    
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
    if batch_size != None:
      assert self._ffconfig.batch_size == batch_size, "batch size is not correct use -b to set it"
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

    assert self._output_tensor != None, "tensor is not init"
    if (isinstance(x, list) == False):
      input_tensors = [x]
    else:
      input_tensors = x
    label_tensor = y
    self._verify_tensors(input_tensors, label_tensor)
    self._create_data_loaders(input_tensors, label_tensor)
    self._ffmodel.init_layers()
    self._train(epochs, callbacks, eval=False)

  def _create_label_tensor(self):
    label_ffhandle = self._ffmodel.label_tensor
    self._label_tensor = Tensor(ffconfig=self._ffconfig, batch_shape=(self._ffconfig.batch_size, 1), dtype=self._label_type)
    self._label_tensor.ffhandle = label_ffhandle

  def _create_input_tensors(self):
    idx = 0
    for input_tensor in self._input_tensors:
      self._input_tensors[idx].create_ff_tensor(self._ffmodel)
      idx += 1
      
  def _create_flexflow_layers(self):
    self._my_onnx_model = ONNXModelKeras(self._onnx_model, self._ffconfig, self._ffmodel)
    input_dict = {}
    for input_tensor in self._input_tensors:
      key = "input_" + str(input_tensor.key)
      input_dict[key] = input_tensor.ffhandle
    self._output_tensor = self._my_onnx_model.apply(self._ffmodel, input_dict)
    
  def _create_optimizer(self):
    assert self._ffoptimizer != None, "optimizer is not set"
    if (isinstance(self._ffoptimizer, ff_keras_optimizer.SGD) == True) or (isinstance(self._ffoptimizer, ff_keras_optimizer.Adam) == True):
      self._ffoptimizer.create_ffhandle(self._ffmodel)
    else:
      assert 0, "unknown optimizer"
      
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
    
  def _create_data_loaders(self, x_trains, y_train):
    # Todo: check all num_samples, should be the same
    input_shape = x_trains[0].shape
    self._num_samples = input_shape[0]

    assert len(self._input_tensors) != 0, "input_tensor is not set"
    assert self._label_tensor != 0, "label_tensor is not set"

    idx = 0
    for x_train in x_trains:
      dataloader = self._ffmodel.create_data_loader(self._input_tensors[idx].ffhandle, x_train)
      self._input_dataloaders.append(dataloader)
      self._input_dataloaders_dim.append(len(input_shape))
      idx += 1
    dataloader = self._ffmodel.create_data_loader(self._label_tensor.ffhandle, y_train)
    self._label_dataloader = dataloader
    self._label_dataloader_dim = len(input_shape)

  def _train(self, epochs, callbacks, eval=False):
    if callbacks != None:
      for callback in callbacks:
        callback.set_model(self)

    if callbacks != None:
      for callback in callbacks:
        callback.on_train_begin()

    ts_start = self._ffconfig.get_current_time()
    epoch = 0
    epoch_flag = True
    self.__tracing_id += 1
    while (epoch < epochs) and (epoch_flag == True):
      if callbacks != None:
        for callback in callbacks:
          callback.on_epoch_begin(epoch)

      for dataloader in self._input_dataloaders:
        dataloader.reset()
      self._label_dataloader.reset()
      self._ffmodel.reset_metrics()
      iterations = self._num_samples / self._ffconfig.batch_size

      for iter in range(0, int(iterations)):
        if callbacks != None:
          for callback in callbacks:
            callback.on_batch_begin(iter)

        for dataloader in self._input_dataloaders:
          dataloader.next_batch(self._ffmodel)
        self._label_dataloader.next_batch(self._ffmodel)

        self._ffconfig.begin_trace(self.__tracing_id)
        self._ffmodel.forward()
        # for layer in self._layers:
        #   layer.ffhandle.forward(self._ffmodel)
        if eval == False:
          self._ffmodel.zero_gradients()
          self._ffmodel.backward()
          self._ffmodel.update()
        else:
          self._ffmodel.compute_metrics()
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
    
class Model(tf_keras_Model):
  def __init__(self, inputs, outputs, name=None):
    super(Model, self).__init__(inputs=inputs, outputs=outputs, name=name)
    
    if (isinstance(inputs, dict) == True):
      onnx_model = keras2onnx.convert_keras(self, name)
      self._base_model = BaseModel(inputs=inputs, onnx_model=onnx_model)

  def compile(self,
              optimizer,
              loss=None,
              metrics=None,
              loss_weights=None,
              weighted_metrics=None,
              run_eagerly=None,
              **kwargs):
    self._base_model.compile(optimizer=optimizer, loss=loss, metrics=metrics, loss_weights=loss_weights, weighted_metrics=weighted_metrics, run_eagerly=run_eagerly, **kwargs)
    
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
    self._base_model.fit(x=x, y=y, batch_size=batch_size, epochs=epochs, verbose=verbose, callbacks=callbacks,
      validation_split=validation_split, validation_data=validation_data, shuffle=shuffle, class_weight=class_weight,
      sample_weight=sample_weight, initial_epoch=initial_epoch, steps_per_epoch=steps_per_epoch, validation_steps=validation_steps,
      validation_batch_size=validation_batch_size, validation_freq=validation_freq, max_queue_size=max_queue_size, workers=workers,
      use_multiprocessing=use_multiprocessing)
      
class Sequential(tf_keras_Model):
  def __init__(self, inputs, outputs, name=None):
    super(Sequential, self).__init__(inputs=inputs, outputs=outputs, name=name)
    
    if (isinstance(inputs, dict) == True):
      onnx_model = keras2onnx.convert_keras(self, name)
      self._base_model = BaseModel(inputs=inputs, onnx_model=onnx_model)

  def compile(self,
              optimizer,
              loss=None,
              metrics=None,
              loss_weights=None,
              weighted_metrics=None,
              run_eagerly=None,
              **kwargs):
    self._base_model.compile(optimizer=optimizer, loss=loss, metrics=metrics, loss_weights=loss_weights, weighted_metrics=weighted_metrics, run_eagerly=run_eagerly, **kwargs)
    
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
    self._base_model.fit(x=x, y=y, batch_size=batch_size, epochs=epochs, verbose=verbose, callbacks=callbacks,
      validation_split=validation_split, validation_data=validation_data, shuffle=shuffle, class_weight=class_weight,
      sample_weight=sample_weight, initial_epoch=initial_epoch, steps_per_epoch=steps_per_epoch, validation_steps=validation_steps,
      validation_batch_size=validation_batch_size, validation_freq=validation_freq, max_queue_size=max_queue_size, workers=workers,
      use_multiprocessing=use_multiprocessing)
