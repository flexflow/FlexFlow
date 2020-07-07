import flexflow.core as ff

class Tensor(object):
  def __init__(self, ffmodel=0, batch_shape=0, name=0, dtype=0, meta_only=False, ffhandle=0):
    self.ffhandle = ffhandle
    self.to_layers = []
    self.from_layer = 0
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
      self.batch_shape = tuple(batch_shape)
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
    if (self.num_dims == 2 or self.num_dims == 4):
      self.ffhandle = ffmodel.create_tensor(self.batch_shape, self.name, self.dtype);
    else:
      assert 0, "un-supported dims"
    self.__verify_ffhandle_shape()
      
  def set_ffhandle(self, ffhandle):
    assert isinstance(ffhandle, ff.Tensor) == True, "[Tensor]: ffhandle is not the correct type"
    assert self.ffhandle == 0, "[Tensor]: check handle, already set"
    self.ffhandle = ffhandle
    if (self.batch_shape[0] == 0):
      self.set_batch_size(ffhandle.dims[0])
    self.__verify_ffhandle_shape()
    
  def set_from_layer(self, layer):
    assert self.from_layer == 0, "[Tensor]: from layer has been set"
    self.from_layer = layer
    
  def set_batch_size(self, size):
    lst = list(self.batch_shape)
    lst[0] = size
    self.batch_shape = tuple(lst)
    
  def __verify_ffhandle_shape(self):
    assert self.num_dims == self.ffhandle.num_dims, "[Tensor]: check tensor shape"
    if (self.batch_shape[0] == 0):
      self.set_batch_size(self.ffhandle.dims[0])
    for i in range(0, self.num_dims):
      assert self.batch_shape[i] == self.ffhandle.dims[i], "[Tensor]: please check shape dim %d (%d == %d)" %(i, self.batch_shape[i], self.ffhandle.dims[i])

class Input(Tensor):
  def __init__(self, shape=None, batch_shape=None,
               name="", dtype=None, sparse=False,
               tensor=None):
    super(Input, self).__init__(0, batch_shape, name, dtype, meta_only=True) 
    
  def set_to_layer(self, layer):
    self.to_layers.append(layer)
    
