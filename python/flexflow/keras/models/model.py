import flexflow.core as ff

from .base_model import BaseModel
from flexflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation

import builtins

def init_internal_model():
  builtins.internal_ffconfig = ff.FFConfig()
  builtins.internal_ffconfig.parse_args()
  builtins.internal_ffmodel = ff.FFModel(builtins.internal_ffconfig)

def delete_internal_model():
  del builtins.internal_ffconfig
  del builtins.internal_ffmodel

class Model(BaseModel):
  def __init__(self, input_tensor, output_tensor):
    super(Model, self).__init__()
    
    self.input_tensor = input_tensor
    self.output_tensor = output_tensor
    
  def add(self, layer):
    self._layers[self._nb_layers] = layer
    layer.layer_id = self._nb_layers
    self._nb_layers += 1
    layer.add_to_model(self.ffmodel)
    
  def add_softmax(self, input_tensor, label_tensor):
    output_tensor = self.ffmodel.softmax("softmax", input_tensor, label_tensor.ffhandle)
    print(self.ffmodel._nb_layers)
    print(self._nb_layers)
    layer = self.ffmodel.get_layer_by_id(self._nb_layers)
    self._layers[self._nb_layers] = layer
    layer.layer_id = self._nb_layers
    self._nb_layers += 1
    
  def fit(self, input_tensor, label_tensor, epochs=1):
    self._create_data_loaders(input_tensor, label_tensor)
    self._set_optimizer()     
    self.ffmodel.init_layers()
    self._train(epochs)
    
  def fit_old(self, dataloader, alexnetconfig):    
    self._set_optimizer() 
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
        #self.ffmodel.forward()
        for layer_id in self._layers:
          layer = self._layers[layer_id]
          layer.forward(self.ffmodel)
        self.ffmodel.zero_gradients()
        self.ffmodel.backward()
        self.ffmodel.update()
        if (epoch > 0):
          self.ffconfig.end_trace(111)

    ts_end = self.ffconfig.get_current_time()
    run_time = 1e-6 * (ts_end - ts_start);
    print("epochs %d, ELAPSED TIME = %.4fs, THROUGHPUT = %.2f samples/s\n" %(epochs, run_time, dataloader.get_num_samples() * epochs / run_time));