import flexflow.core as ff

import builtins

class Input(object):
  def __init__(self, shape=None, batch_shape=None,
               name=None, dtype=None, sparse=False,
               tensor=None):
    if (dtype == "float32"):
      self.dtype = ff.DataType.DT_FLOAT
    elif (dtype == "float64"):
      self.dtype = ff.DataType.DT_DOUBLE
    elif (dtype == "int32"):
      self.dtype = ff.DataType.DT_INT32
    elif (dtype == "int64"):
      self.dtype = ff.DataType.DT_INT64
    else:
      assert 0, "not supported"
    self.batch_shape = batch_shape
    self.handle = 0
    self.num_dims = len(batch_shape)
    self.name = name
    self.__create_ff_tensor()
    
  def __create_ff_tensor(self):
    if (self.num_dims == 2):
      self.handle = builtins.internal_ffmodel.create_tensor_2d(self.batch_shape, "", self.dtype);
    elif (self.num_dims == 4):
      self.handle = builtins.internal_ffmodel.create_tensor_4d(self.batch_shape, "", self.dtype);
    else:
      assert 0, "un-supported dims"
    