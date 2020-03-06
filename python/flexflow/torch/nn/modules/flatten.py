class Flatten(object):
  def __init__(self, start_dim=1, end_dim=-1):
    super(Flatten, self).__init__()
    self.start_dim = start_dim
    self.end_dim = end_dim
