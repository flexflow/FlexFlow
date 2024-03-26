#include "runtime/config.h"
#include "utils/exception.h"
#include "utils/parse.h"
namespace FlexFlow {

// issue:https://github.com/flexflow/FlexFlow/issues/942
void FFConfig::parse_args(char **argv, int argc) {
  constexpr size_t argv_length = sizeof(argv) / sizeof(argv[0]);
  ArgsParser args;
  auto epochs_ref = add_optional_argument(
      args, "--epochs", std::optional<int>(1), "Number of epochs.");
  auto batch_size_ref =
      add_optional_argument(args,
                            "--batch-size",
                            std::optional<int>(32),
                            "Size of each batch during training");
  auto numnodes_ref = add_optional_argument(
      args, "--num-nodes", std::optional<int>(1), "Number of nodes");
  auto ll_cpu_ref = add_required_argument(
      args, "-ll:cpu", std::optional<int>(1), "CPUs per node");
  auto ll_gpu_ref = add_required_argument(args,
                                          "-ll:gpu",
                                          std::optional<int>(0),
                                          "GPUs per node"); // workersPerNode

  auto learning_rate_ref =
      add_optional_argument(args,
                            "--learning-rate",
                            std::optional<float>(0.01f),
                            "Learning rate for the optimizer");

  auto weight_decay_ref =
      add_optional_argument(args,
                            "--weight-decay",
                            std::optional<float>(0.0001f),
                            "Weight decay for the optimizer");

  auto profile_ref = add_optional_argument(
      args, "--profile", std::optional<bool>(false), "Enable profiling");

  auto perform_fusion_ref = add_optional_argument(
      args, "--fusion", std::optional<bool>(false), "Enable fusion");

  auto simulator_work_space_size_ref =
      add_optional_argument(args,
                            "--simulator-work-space-size",
                            std::optional<size_t>(0),
                            "Simulator workspace size");

  auto search_budget_ref = add_optional_argument(
      args, "--search-budget", std::optional<int>(0), "Search budget");

  auto search_alpha_ref = add_optional_argument(
      args, "--search-alpha", std::optional<float>(0.0f), "Search alpha");

  auto search_overlap_backward_update_ref = add_optional_argument(
      args, "--overlap", std::optional<bool>(false), "Enable overlap");

  auto only_data_parallel_ref =
      add_optional_argument(args,
                            "--only-data-parallel",
                            std::optional<bool>(false),
                            "Only use data parallelism");

  auto enable_parameter_parallel_ref =
      add_optional_argument(args,
                            "--enable-parameter-parallel",
                            std::optional<bool>(false),
                            "Enable parameter parallelism");

  auto enable_inplace_optimizations_ref =
      add_optional_argument(args,
                            "--enable-inplace-optimizations",
                            std::optional<bool>(false),
                            "Enable inplace optimizations");

  auto allow_tensor_op_math_conversion_ref =
      add_optional_argument(args,
                            "--allow-tensor-op-math-conversion",
                            std::optional<bool>(false),
                            "Allow tensor op math conversion");

  auto dataset_path_ref = add_optional_argument(args,
                                                "--dataset-path",
                                                std::optional<std::string>(""),
                                                "Path to the dataset");

  auto export_strategy_computation_graph_file_ref =
      add_optional_argument(args,
                            "--taskgraph",
                            std::optional<std::string>(""),
                            "Export strategy computation graph file");

  auto include_costs_dot_graph_ref =
      add_optional_argument(args,
                            "--include-costs-dot-graph",
                            std::optional<bool>(false),
                            "Include costs dot graph");

  auto substitution_json_ref =
      add_optional_argument(args,
                            "--substitution-json",
                            std::optional<std::string>(""),
                            "Substitution json path");

  auto machine_model_version_ref =
      add_optional_argument(args,
                            "--machine-model-version",
                            std::optional<int>(0),
                            "Machine model version");

  auto machine_model_file_ref =
      add_optional_argument(args,
                            "--machine-model-file",
                            std::optional<std::string>(""),
                            "Machine model file");

  auto simulator_segment_size_ref =
      add_optional_argument(args,
                            "--simulator-segment-size",
                            std::optional<int>(0),
                            "Simulator segment size");

  auto simulator_max_num_segments_ref =
      add_optional_argument(args,
                            "--simulator-max-num-segments",
                            std::optional<int>(0),
                            "Simulator max number of segments");

  auto search_num_nodes_ref = add_optional_argument(args,
                                                    "--search-num-nodes",
                                                    std::optional<int>(0),
                                                    "Search number of nodes");

  auto search_num_workers_ref =
      add_optional_argument(args,
                            "--search-num-workers",
                            std::optional<int>(0),
                            "Search number of workers");

  auto base_optimize_threshold_ref =
      add_optional_argument(args,
                            "--base-optimize-threshold",
                            std::optional<int>(0),
                            "Base optimize threshold");

  auto enable_control_replication_ref =
      add_optional_argument(args,
                            "--enable-control-replication",
                            std::optional<bool>(false),
                            "Enable control replication");

  /*auto ll_csize_ref = add_required_argument(args,"-ll:csize",
  std::optional<int>(1024), "size of CPU DRAM memory per process(in MB)");

  auto ll_gsize_ref = add_required_argument(args,"-ll:gsize",
  std::optional<int>(0), "size of GPU DRAM memory per process");

  auto ll_rsize_ref = add_required_argument(args,"-ll:rsize",
  std::optional<int>(0), "size of GASNet registered RDMA memory available per
  process (in MB)");

  auto ll_fsize_ref = add_required_argument(args,"-ll:fsize",
  std::optional<int>(1), "size of framebuffer memory for each GPU (in MB)");

  auto ll_zsize_ref = add_required_argument(args,"-ll:zsize",
  std::optional<int>(0), "size of zero-copy memory for each GPU (in MB)");

  auto lg_window_ref = add_required_argument(args,"-lg:window",
  std::optional<int>(8192), "maximum number of tasks that can be created in a
  parent task window");

  auto lg_sched_ref = add_required_argument(args,"-lg:sched",
  std::optional<int>(1024), " minimum number of tasks to try to schedule for
  each invocation of the scheduler");
  */
  ArgsParser result =
      parse_args(args, argv_length, const_cast<char const **>(argv));

  epochs = get(result, epochs_ref);
  batchSize = get(result, batch_size_ref);
  numNodes = get(result, numnodes_ref);
  cpusPerNode = get(result, ll_cpu_ref);
  workersPerNode = get(result, ll_gpu_ref);
  learningRate = get(result, learning_rate_ref);
  weightDecay = get(result, weight_decay_ref);
  profiling = get(result, profile_ref);
  perform_fusion = get(result, perform_fusion_ref);
  simulator_work_space_size = get(result, simulator_work_space_size_ref);
  search_budget = get(result, search_budget_ref);
  search_alpha = get(result, search_alpha_ref);
  search_overlap_backward_update =
      get(result, search_overlap_backward_update_ref);
  only_data_parallel = get(result, only_data_parallel_ref);
  enable_parameter_parallel = get(result, enable_parameter_parallel_ref);
  enable_inplace_optimizations = get(result, enable_inplace_optimizations_ref);
  allow_tensor_op_math_conversion =
      get(result, allow_tensor_op_math_conversion_ref);
  dataset_path = get(result, dataset_path_ref);
  export_strategy_computation_graph_file =
      get(result, export_strategy_computation_graph_file_ref);
  include_costs_dot_graph = get(result, include_costs_dot_graph_ref);
  substitution_json_path = get(result, substitution_json_ref);
  machine_model_version = get(result, machine_model_version_ref);
  machine_model_file = get(result, machine_model_file_ref);
  simulator_segment_size = get(result, simulator_segment_size_ref);
  simulator_max_num_segments = get(result, simulator_max_num_segments_ref);
  search_num_nodes = get(result, search_num_nodes_ref);
  search_num_workers = get(result, search_num_workers_ref);
  base_optimize_threshold = get(result, base_optimize_threshold_ref);
  enable_control_replication = get(result, enable_control_replication_ref);
  /*ll_util = get(result, ll_util_ref);
  ll_csize = get(result, ll_csize_ref);
  ll_gsize = get(result, ll_gsize_ref);
  ll_rsize = get(result, ll_rsize_ref);
  ll_fsize = get(result, ll_fsize_ref);
  ll_zsize = get(result, ll_zsize_ref);
  lg_window = get(result, lg_window_ref);
  lg_sched = get(result, lg_sched_ref);*/
}

} // namespace FlexFlow
