class Op(object):
  def __init__(self):
    self.layer_id = -1
    self._handle = 0
    self._ffmodel = 0
    
  def set_flexflow_model(self, model):
    self._ffmodel = model