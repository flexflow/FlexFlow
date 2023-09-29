#include "runtime/config.h"
#include "utils/parse.h"

namespace FlexFlow {

void FFConfig::parse_args(char **argv, int argc) {
  ArgsParser args;
  auto epochs_ref = args.add_argument("--epochs", 1, "Number of epochs.");
  auto batch_size_ref = args.add_argument(
      "--batch-size", 32, "Size of each batch during training");
  auto learning_rate_ref = args.add_argument(
      "--learning-rate", 0.01f, "Learning rate for the optimizer");
  auto weight_decay_ref = args.add_argument(
      "--weight-decay", 0.0001f, "Weight decay for the optimizer");
  auto dataset_pat_ref =
      args.add_argument("--dataset-path", "", "Path to the dataset");
  auto search_budget_ref =
      args.add_argument("--search-budget", 0, "Search budget");
  auto search_alpha_ref =
      args.add_argument("--search-alpha", 0.0f, "Search alpha");
  auto simulator_workspace_size_ref = args.add_argument(
      "--simulator-workspace-size", 0, "Simulator workspace size");
  auto only_data_parallel_ref = args.add_argument(
      "--only-data-parallel", false, "Only use data parallelism");
  auto enable_parameter_parallel = args.add_argument(
      "--enable-parameter-parallel", false, "Enable parameter parallelism");
  auto nodes_ref = args.add_argument("--nodes", 1, "Number of nodes");
  auto profiling_ref =
      args.add_argument("--profiling", false, "Enable profiling");
  auto allow_tensor_op_math_conversion_ref =
      args.add_argument("--allow-tensor-op-math-conversion",
                        false,
                        "Allow tensor op math conversion");
  auto fustion_ref = args.add_argument("--fusion", false, "Enable fusion");
  auto overlap_ref = args.add_argument("--overlap", false, "Enable overlap");
  auto taskgraph_ref = args.add_argument(
      "--taskgraph", "", "Export strategy computation graph file");
  auto = include_costs_dot_graph_ref = args.add_argument(
      "--include-costs-dot-graph", false, "Include costs dot graph");
  auto machine_model_version_ref =
      args.add_argument("--machine-model-version", 0, "Machine model version");
  auto machine_model_file_ref =
      args.add_argument("--machine-model-file", "", "Machine model file");
  auto simulator_segment_size_ref = args.add_argument(
      "--simulator-segment-size", 0, "Simulator segment size");
  auto simulator_max_num_segments_ref = args.add_argument(
      "--simulator-max-num-segments", 0, "Simulator max number of segments");
  auto enable_inplace_optimizations_ref = args.add_argument(
      "--enable-inplace-optimizations", false, "Enable inplace optimizations");
  auto search_num_nodes_ref =
      args.add_argument("--search-num-nodes", 0, "Search number of nodes");
  auto search_num_workers_ref =
      args.add_argument("--search-num-workers", 0, "Search number of workers");
  auto base_optimize_threshold_ref = args.add_argument(
      "--base-optimize-threshold", 0, "Base optimize threshold");
  auto enable_control_replication_ref = args.add_argument(
      "--enable-control-replication", false, "Enable control replication");
  auto python_data_loader_type_ref = args.add_argument(
      "--python-data-loader-type", 0, "Python data loader type");
  auto substitution_json_ref =
      args.add_argument("--substitution-json", "", "Substitution json path");

  // legion arguments
  auto level_ref = args.add_argument("-level", 5, "level of logging output");
  auto logfile_ref = args.add_argument("-logfile", "", "name of log file");
  auto ll_cpu_ref = args.add_argument("-ll:cpu", 1, "CPUs per node");
  auto ll_gpu_ref = args.add_argument("-ll:gpu", 0, "GPUs per node");
  auto ll_util_ref = args.add_argument(
      "-ll:util", 1, "utility processors to create per process");
  auto ll_csize_ref = args.add_argument(
      "-ll:csize", 1024, "size of CPU DRAM memory per process(in MB)");
  auto ll_gsize_ref =
      args.add_argument("-ll:gsize", 0, "size of GPU DRAM memory per process");
  auto ll_rsize_ref = args.add_argument(
      "-ll:rsize",
      0,
      "size of GASNet registered RDMA memory available per process (in MB)");
  auto ll_fsize_ref = args.add_argument(
      "-ll:fsize", 1, "size of framebuffer memory for each GPU (in MB)");
  auto ll_zsize_ref = args.add_argument(
      "-ll:zsize", 0, "size of zero-copy memory for each GPU (in MB)");
  auto lg_window_ref = args.add_argument(
      "-lg:window",
      8192,
      "maximum number of tasks that can be created in a parent task window");
  auto lg_sched_ref =
      args.add_argument("-lg:sched",
                        1024,
                        " minimum number of tasks to try to schedule for each "
                        "invocation of the scheduler");

  args.parse_args(argc, argv);

  batch_size = args.get(batch_size_ref) epochs = args.get(epochs_ref);
  learning_rate = args.get(learning_rate_ref);
  weight_decay = args.get(weight_decay_ref);
  dataset_path = args.get(dataset_pat_ref);
  search_budget = args.get(search_budget_ref);
  search_alpha = args.get(search_alpha_ref);
  simulator_work_space_size = args.get(simulator_workspace_size_ref));
  only_data_parallel = args.get(only_data_parallel_ref);
  enable_parameter_parallel = args.get(enable_parameter_parallel);
  numNodes = args.get(nodes_ref);
  profiling = args.get(profiling_ref);
  allow_tensor_op_math_conversion =
      args.get(allow_tensor_op_math_conversion_ref);
  perform_fusion = args.get(fustion_ref);
  search_overlap_backward_update = args.get(overlap_ref);
  export_strategy_computation_graph_file = args.get(task_graph_ref);
  include_costs_dot_graph = args.get(include_costs_dot_graph_ref);
  machine_model_version = args.get(machine_model_version_ref);
  machine_model_file = args.get(machine_model_file_ref);
  simulator_segment_size = args.get(simulator_segment_size_ref);
  simulator_max_num_segments = args.get(simulator_max_num_segments_ref);
  enable_inplace_optimizations = args.get(enable_inplace_optimizations_ref);
  search_num_nodes = args.get(search_num_nodes_ref);
  search_num_workers = args.get(search_num_workers_ref);
  base_optimize_threshold = args.get(base_optimize_threshold_ref);
  enable_control_replication = args.get(enable_control_replication_ref);
  python_data_loader_type = args.get(python_data_loader_type_ref);
  substitution_json_path = args.get(substitution_json_ref);

  // legion arguments
  cpusPerNode = args.get(ll_cpu_ref);
  workersPerNode = args.get(ll_gpu_ref);
}

} // namespace FlexFlow
