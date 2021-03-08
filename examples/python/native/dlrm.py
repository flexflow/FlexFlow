from flexflow.core import *

def top_level_task():
  ffconfig = FFConfig()
  dlrmconfig = DLRMConfig()
  print("Python API batchSize(%d) workersPerNodes(%d) numNodes(%d)" %(ffconfig.batch_size, ffconfig.workers_per_node, ffconfig.num_nodes))
  print(dlrmconfig.dataset_path, dlrmconfig.arch_interaction_op)
  print(dlrmconfig.sparse_feature_size, dlrmconfig.sigmoid_bot, dlrmconfig.sigmoid_top, dlrmconfig.embedding_bag_size, dlrmconfig.loss_threshold)
  print(dlrmconfig.mlp_bot)
  print(dlrmconfig.mlp_top)
  print(dlrmconfig.embedding_size)
  ffmodel = FFModel(ffconfig)


if __name__ == "__main__":
  print("dlrm")
  top_level_task()
