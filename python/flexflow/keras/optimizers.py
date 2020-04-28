import flexflow.core as ff

class SGD(object):
  def __init__(self, learning_rate=0.01):
    self.learning_rate = learning_rate
    self.handle = 0