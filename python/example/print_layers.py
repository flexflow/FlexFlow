from flexflow.core import *

def top_level_task():
  ffconfig = FFConfig()
  ffconfig.parse_args()
  print("Python API batchSize(%d) workersPerNodes(%d) numNodes(%d)" %(ffconfig.get_batch_size(), ffconfig.get_workers_per_node(), ffconfig.get_num_nodes()))
  ffmodel = FFModel(ffconfig)
  
  dims = [ffconfig.get_batch_size(), 3, 229, 229]
  #print(dims)
  input = ffmodel.create_tensor_4d(dims, "", DataType.DT_FLOAT);
  
  dims_label = [ffconfig.get_batch_size(), 1]
  #print(dims)
  label = ffmodel.create_tensor_2d(dims_label, "", DataType.DT_FLOAT);
  
  t = ffmodel.conv2d("conv1", input, 64, 11, 11, 4, 4, 2, 2)
  
  
  # Data Loader
  dataloader = DataLoader(ffmodel, input, label)
  
  ffmodel.init_layers()
  ffmodel.print_layers()

if __name__ == "__main__":
  print("alexnet")
  top_level_task()
