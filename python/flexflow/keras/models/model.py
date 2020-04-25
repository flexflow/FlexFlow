import flexflow.core as ff

from flexflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation

import builtins

builtins.internal_ffconfig = ff.FFConfig()
builtins.internal_ffconfig.parse_args()
builtins.internal_ffmodel = ff.FFModel(builtins.internal_ffconfig)

class Model(object):
  def __init__(self, input_tensor, output_tensor):
    self.ffconfig = ff.FFConfig()
    self.ffconfig.parse_args()
    print("Python API batchSize(%d) workersPerNodes(%d) numNodes(%d)" %(self.ffconfig.get_batch_size(), self.ffconfig.get_workers_per_node(), self.ffconfig.get_num_nodes()))
    self.ffmodel = ff.FFModel(self.ffconfig)
    
    self.input_tensor = input_tensor
    self.output_tensor = output_tensor