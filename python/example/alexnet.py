from flexflow.core import *

class FFTest(object):
  def __init__(self):
    print("FFTest Constructor");
    
  def __del__(self):
    print("FFTest Destructor")

def top_level_task():
  ffconfig = FFConfig()
  ffconfig.parse_args()
  print("Python API batchSize(%d) workersPerNodes(%d) numNodes(%d)" %(ffconfig.get_batch_size(), ffconfig.get_workers_per_node(), ffconfig.get_num_nodes()))
  ffmodel = FFModel(ffconfig)

if __name__ == "__main__":
  print("alexnet")
  top_level_task()