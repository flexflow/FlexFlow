#include "config.h"
#include "utils/parser.h"

namespace FlexFlow {

// void FFConfig::parse_args(char **argv, int argc) {
//   ArgsParser args;
//   args.add_argument("--epochs", 1, "Number of epochs.");
//   args.add_argument("--batch-size", 32, "Size of each batch during training");
//   args.add_argument(
//       "--learning-rate", 0.01f, "Learning rate for the optimizer");
//   args.add_argument(
//       "--weight-decay", 0.0001f, "Weight decay for the optimizer");
//   args.add_argument("--dataset-path", "", "Path to the dataset");
//   args.add_argument("--search-budget", 0, "Search budget");
//   args.add_argument("--search-alpha", 0.0f, "Search alpha");
//   args.add_argument(
//       "--simulator-workspace-size", 0, "Simulator workspace size");
//   args.add_argument("--only-data-parallel", false, "Only use data parallelism");
//   args.add_argument(
//       "--enable-parameter-parallel", false, "Enable parameter parallelism");
//   args.add_argument("--nodes", 1, "Number of nodes");
//   args.add_argument("--profiling", false, "Enable profiling");
//   args.add_argument("--allow-tensor-op-math-conversion",
//                     false,
//                     "Allow tensor op math conversion");
//   args.add_argument("--fusion", false, "Enable fusion");
//   args.add_argument("--overlap", false, "Enable overlap");
//   args.add_argument(
//       "--taskgraph", "", "Export strategy computation graph file");
//   args.add_argument(
//       "--include-costs-dot-graph", false, "Include costs dot graph");
//   args.add_argument("--machine-model-version", 0, "Machine model version");
//   args.add_argument("--machine-model-file", "", "Machine model file");
//   args.add_argument("--simulator-segment-size", 0, "Simulator segment size");
//   args.add_argument(
//       "--simulator-max-num-segments", 0, "Simulator max number of segments");
//   args.add_argument(
//       "--enable-inplace-optimizations", false, "Enable inplace optimizations");
//   args.add_argument("--search-num-nodes", 0, "Search number of nodes");
//   args.add_argument("--search-num-workers", 0, "Search number of workers");
//   args.add_argument("--base-optimize-threshold", 0, "Base optimize threshold");
//   args.add_argument(
//       "--enable-control-replication", false, "Enable control replication");
//   args.add_argument("--python-data-loader-type", 0, "Python data loader type");
//   args.add_argument("--substitution-json", "", "Substitution json path");

//   // legion arguments
//   args.add_argument("-level", 5, "level of logging output");
//   args.add_argument("-logfile", "", "name of log file");
//   args.add_argument("-ll:cpu", 1, "CPUs per node");
//   args.add_argument("-ll:gpu", 0, "GPUs per node");
//   args.add_argument("-ll:util", 1, "utility processors to create per process");
//   args.add_argument(
//       "-ll:csize", 1024, "size of CPU DRAM memory per process(in MB)");
//   args.add_argument("-ll:gsize", 0, "size of GPU DRAM memory per process");
//   args.add_argument(
//       "-ll:rsize",
//       0,
//       "size of GASNet registered RDMA memory available per process (in MB)");
//   args.add_argument(
//       "-ll:fsize", 1, "size of framebuffer memory for each GPU (in MB)");
//   args.add_argument(
//       "-ll:zsize", 0, "size of zero-copy memory for each GPU (in MB)");
//   args.add_argument(
//       "-lg:window",
//       8192,
//       "maximum number of tasks that can be created in a parent task window");
//   args.add_argument("-lg:sched",
//                     1024,
//                     " minimum number of tasks to try to schedule for each "
//                     "invocation of the scheduler");

//   args.parse_args(argc, argv);

//   batch_size = args.get<int>("batch-size");
//   epochs = args.get<int>("epochs");
//   learning_rate = args.get<float>("learning-rate");
//   weight_decay = args.get<float>("weight-decay");
//   dataset_path = args.get<std::string>("dataset-path");
//   search_budget = args.get<size_t>("search-budget");
//   search_alpha = args.get<float>("search-alpha");
//   simulator_work_space_size = args.get<size_t>("simulator-workspace-size");
//   only_data_parallel = args.get<bool>("only-data-parallel");
//   enable_parameter_parallel = args.get<bool>("enable-parameter-parallel");
//   numNodes = args.get<int>("nodes");
//   profiling = args.get<bool>("profiling");
//   allow_tensor_op_math_conversion =
//       args.get<bool>("allow-tensor-op-math-conversion");
//   perform_fusion = args.get<bool>("fusion");
//   search_overlap_backward_update = args.get<bool>("overlap");
//   export_strategy_computation_graph_file = args.get<std::string>("--taskgraph");
//   include_costs_dot_graph = args.get<bool>("include-costs-dot-graph");
//   machine_model_version = args.get<int>("machine-model-version");
//   machine_model_file = args.get<std::string>("machine-model-file");
//   simulator_segment_size = args.get<int>("simulator-segment-size");
//   simulator_max_num_segments = args.get<int>("simulator-max-num-segments");
//   enable_inplace_optimizations = args.get<bool>("enable-inplace-optimizations");
//   search_num_nodes = args.get<int>("search-num-nodes");
//   search_num_workers = args.get<int>("search-num-workers");
//   base_optimize_threshold = args.get<int>("base-optimize-threshold");
//   enable_control_replication = args.get<bool>("enable-control-replication");
//   python_data_loader_type = args.get<int>("python-data-loader-type");
//   substitution_json_path = args.get<std::string>("substitution-json");

//   // legion arguments
//   cpusPerNode = args.get<int>("-ll:cpu");
//   workersPerNode = args.get<int>("-ll:gpu");
// }

}  // namespace FlexFlow 