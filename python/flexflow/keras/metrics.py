import flexflow.core as ff

class Metric(object):
  def __init__(self, name=None, dtype=None, **kwargs):
    self.name = name
    self.dtype = dtype
    self.type = None
    
class Accuracy(Metric):
  def __init__(self, 
               name='accuracy', 
               dtype=None):
    super(Accuracy, self).__init__(name=name, dtype=dtype)
    self.type = ff.MetricsType.METRICS_ACCURACY
    
class CategoricalCrossentropy(Metric):
  def __init__(self, 
               name='categorical_crossentropy', 
               dtype=None,
               from_logits=False,
               label_smoothing=0):
    super(CategoricalCrossentropy, self).__init__(name=name, dtype=dtype)
    self.type = ff.MetricsType.METRICS_CATEGORICAL_CROSSENTROPY
    
class SparseCategoricalCrossentropy(Metric):
  def __init__(self, 
               name='sparse_categorical_crossentropy', 
               dtype=None,
               from_logits=False,
               axis=1):
    super(SparseCategoricalCrossentropy, self).__init__(name=name, dtype=dtype)
    self.type = ff.MetricsType.METRICS_SPARSE_CATEGORICAL_CROSSENTROPY
    
class MeanSquaredError(Metric):
  def __init__(self, 
               name='mean_squared_error', 
               dtype=None):
    super(MeanSquaredError, self).__init__(name=name, dtype=dtype)
    self.type = ff.MetricsType.METRICS_MEAN_SQUARED_ERROR
    
class RootMeanSquaredError(Metric):
  def __init__(self, 
               name='root_mean_squared_error', 
               dtype=None):
    super(RootMeanSquaredError, self).__init__(name=name, dtype=dtype)
    self.type = ff.MetricsType.METRICS_ROOT_MEAN_SQUARED_ERROR
    
class MeanAbsoluteError(Metric):
  def __init__(self, 
               name='mean_absolute_error', 
               dtype=None):
    super(MeanAbsoluteError, self).__init__(name=name, dtype=dtype)
    self.type = ff.MetricsType.METRICS_MEAN_ABSOLUTE_ERROR

  