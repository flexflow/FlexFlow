import flexflow.core as ff

class Metric(object):
  def __init__(self, name=None, dtype=None, **kwargs):
    self.name = name
    self.dtype = dtype
    
class Accuracy(Metric):
  def __init__(self, 
               name='accuracy', 
               dtype=None):
    super(Accuracy, self).__init__(name=name, dtype=dtype)
    
class CategoricalCrossentropy(Metric):
  def __init__(self, 
               name='categorical_crossentropy', 
               dtype=None,
               from_logits=False,
               label_smoothing=0):
    super(CategoricalCrossentropy, self).__init__(name=name, dtype=dtype)
    
class SparseCategoricalCrossentropy(Metric):
  def __init__(self, 
               name='sparse_categorical_crossentropy', 
               dtype=None,
               from_logits=False,
               axis=1):
    super(CategoricalCrossentropy, self).__init__(name=name, dtype=dtype)
    
class MeanSquaredError(Metric):
  def __init__(self, 
               name='mean_squared_error', 
               dtype=None):
    super(MeanSquaredError, self).__init__(name=name, dtype=dtype)
    
class RootMeanSquaredError(Metric):
  def __init__(self, 
               name='root_mean_squared_error', 
               dtype=None):
    super(RootMeanSquaredError, self).__init__(name=name, dtype=dtype)
    
class MeanAbsoluteError(Metric):
  def __init__(self, 
               name='mean_absolute_error', 
               dtype=None):
    super(MeanAbsoluteError, self).__init__(name=name, dtype=dtype)

  