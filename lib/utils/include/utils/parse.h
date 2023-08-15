#ifndef _FLEXFLOW_UTILS_INCLUDE_UTILS_PARSE_H
#define _FLEXFLOW_UTILS_INCLUDE_UTILS_PARSE_H

#include "runtime/config.h"
#include <argparse/argparse.hpp>

namespace FlexFlow {

void FFConfig::parse_args(char **argv, int argc) {
  argparse::ArgumentParser args("FFConfig::parse_args");
  args.add_argument("--epochs").default_value(1).help("Number of epochs.");
  args.add_argument("--batch-size").default_value(32).help("Batch size.");
  args.add_argument("--learning-rate")
      .default_value((float)0.01)
      .help("Learning rate.");
  args.add_argument("--weight-decay")
      .default_value((float)0.0001)
      .help("Weight decay.");
  args.add_argument("--dataset").default_value("").help("Dataset path.");
  args.add_argument("--search-budget")
      .default_value(size_t(-1))
      .help("Search budget.");
  args.add_argument("--search-alpha")
      .default_value(float(1.2))
      .help("Search alpha.");
  args.add_argument("--simulator-workspace-size")
      .default_value((size_t)2 * 1024 * 1024 * 1024;)
      .help("Simulator workspace size.");
  args.add_argument("--only-data-parallel")
      .default_value(false)
      .help("Only data parallel.");
  args.add_argument("--enable-parameter-parallel")
      .default_value(false)
      .help("Enable parameter parallel.");
  args.add_argument("--ll:gpu").default_value(1).help("Number of workers.");
  args.add_argument("--ll:cpu").default_value(1).help("Number of cpus.");
  args.add_argument("--nodes").default_value(1).help("Number of nodes.");
  args.add_argument("--profiling")
      .default_value(false)
      .help("Enable profiling.");
  arg.add_argument("--allow-tensor-op-math-conversion")
      .default_value(false)
      .help("Allow tensor op math conversion.");
  args.add_argument("--fusion").default_value(false).help("Enable fusion.");
  args.add_argument("--overlap")
      .default_value(false)
      .help("Search overlap backward update.");
  args.add_argument("--taskgraph")
      .default_value("")
      .help(" export_strategy_task_graph_file");
  args.add_argument("--include-costs-dot-graph")
      .default_value(false)
      .help("Include costs dot graph.");
  args.add_argument("--machine-model-version")
      .default_value(0)
      .help("Machine model version.");
  args.add_argument("--machine-model-file")
      .default_value("")
      .help("Machine model file.");
  args.add_argument("--simulator-segment-size")
      .default_value(16777216)
      .help("Simulator segment size.");
  args.add_argument("--simulator-max-num-segments")
      .default_value(1)
      .help("Simulator max number of segments.");
  args.add_argument("--enable-inplace-optimizations")
      .default_value(false)
      .help("Enable inplace optimizations.");
  args.add_argument("--search-num-nodes")
      .default_value(-1)
      .help("Search number of nodes.");
  args.add_argument("--search-num-workers")
      .default_value(-1)
      .help("Search number of workers.");
  args.add_argument("--base-optimize-threshold")
      .default_value(10)
      .help("Base optimize threshold.");
  args.add_argument("--enable-control-replication")
      .default_value(true)
      .help("Enable control replication.");
  args.add_argument("--python-data-loader-type")
      .default_value(2)
      .help("Python data loader type.");
  args.add_argument("--substitution-json")
      .default_value("")
      .help("Substitution json path.");
  args.parse_args(argc, argv);
  std::cout << "args:" << args << std::endl;

  batch_size = args.get<int>("--batch-size");
  epochs = args.get<int>("--epochs");
  learning_rate = args.get<float>("--learning-rate");
  weight_decay = args.get<float>("--weight-decay");
  dataset_path = args.get<std::string>("--dataset-path");
  search_budget = args.get<size_t>("--search-budget");
  search_alpha = args.get<float>("--search-alpha");
  simulator_work_space_size = args.get<size_t>("--simulator-workspace-size");
  only_data_parallel = args.get<bool>("--only-data-parallel");
  enable_parameter_parallel = args.get<bool>("--enable-parameter-parallel");
  workersPerNode = args.get<int>("--ll:gpu");
  numNodes = args.get<int>("--nodes");
  cpusPerNode = args.get<int>("--ll:cpu");
  profiling = args.get<bool>("--profiling");
  allow_tensor_op_math_conversion =
      args.get<bool>("--allow-tensor-op-math-conversion");
  perform_fusion = args.get<bool>("--fusion");
  search_overlap_backward_update = args.get<bool>("--overlap");
  export_strategy_computation_graph_file = args.get<std::string>("--taskgraph");
  include_costs_dot_graph = args.get<bool>("--include-costs-dot-graph");
  machine_model_version = args.get<int>("--machine-model-version");
  machine_model_file = args.get<std::string>("--machine-model-file");
  simulator_segment_size = args.get<int>("--simulator-segment-size");
  simulator_max_num_segments = args.get<int>("--simulator-max-num-segments");
  enable_inplace_optimizations =
      args.get<bool>("--enable-inplace-optimizations");
  search_num_nodes = args.get<int>("--search-num-nodes");
  search_num_workers = args.get<int>("--search-num-workers");
  base_optimize_threshold = args.get<int>("--base-optimize-threshold");
  enable_control_replication = args.get<bool>("--enable-control-replication");
  python_data_loader_type = args.get<int>("--python-data-loader-type");
  substitution_json_path = args.get<std::string>("--substitution-json");
}

} // namespace FlexFlow

#endif
