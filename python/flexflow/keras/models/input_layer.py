import flexflow.core as ff

class Tensor(object):
  def __init__(self, ffmodel=0, batch_shape=0, name=0, dtype=0, meta_only=False, ffhandle=0):
    self.ffhandle = ffhandle
    self.input_layers = []
    self.output_layer = 0
    if (dtype == "float32" or dtype == ff.DataType.DT_FLOAT):
      self.dtype = ff.DataType.DT_FLOAT
    elif (dtype == "float64" or dtype == ff.DataType.DT_DOUBLE):
      self.dtype = ff.DataType.DT_DOUBLE
    elif (dtype == "int32" or dtype == ff.DataType.DT_INT32):
      self.dtype = ff.DataType.DT_INT32
    elif (dtype == "int64" or dtype == ff.DataType.DT_INT64):
      self.dtype = ff.DataType.DT_INT64
    else:
      assert 0, "not supported"
    # create a tensor
    if (ffhandle == 0):
      self.batch_shape = batch_shape
      self.name = name
      self.num_dims = len(batch_shape)
      if (meta_only == False):
        self.create_ff_tensor(ffmodel)
    # init from handle
    else:
      self.name = ""
      self.num_dims = ffhandle.num_dims
      self.batch_shape = ffhandle.dims
    
  def create_ff_tensor(self, ffmodel):
    if (self.num_dims == 2):
      self.ffhandle = ffmodel.create_tensor_2d(self.batch_shape, self.name, self.dtype);
    elif (self.num_dims == 4):
      self.ffhandle = ffmodel.create_tensor_4d(self.batch_shape, self.name, self.dtype);
    else:
      assert 0, "un-supported dims"
      
  def set_ffhandle(self, ffhandle):
    assert self.ffhandle == 0, "check handle, already set"
    self.ffhandle = ffhandle
    assert self.num_dims == ffhandle.num_dims, "check tensor shape"
    if (self.num_dims == 2):
      assert self.batch_shape[1] == ffhandle.dims[1]
    elif (self.num_dims == 4):
      assert self.batch_shape[1] == ffhandle.dims[1]
      assert self.batch_shape[2] == ffhandle.dims[2]
      assert self.batch_shape[3] == ffhandle.dims[3]
    
  def set_output_layer(self, layer):
    assert self.output_layer == 0, "output layer has been set"
    self.output_layer = layer
    
  def set_batch_size(self, size):
    self.batch_shape[0] = size

class Input(Tensor):
  def __init__(self, shape=None, batch_shape=None,
               name="", dtype=None, sparse=False,
               tensor=None):
    super(Input, self).__init__(0, batch_shape, name, dtype, meta_only=True) 
    
  def set_input_layer(self, layer):
    self.input_layers.append(layer)
    