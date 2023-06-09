#include "ffr/config.h"

void parse(char **argv, int argc, FFConfig &ffconfig) {
  for (int i = 1; i < argc; i++) {
    if ((!strcmp(argv[i], "-e")) || (!strcmp(argv[i], "--epochs"))) {
      ffconfig.epochs = atoi(argv[++i]);
      continue;
    }
    if ((!strcmp(argv[i], "-b")) || (!strcmp(argv[i], "--batch-size"))) {
      ffconfig.batchSize = atoi(argv[++i]);
      continue;
    }
    if ((!strcmp(argv[i], "--lr")) || (!strcmp(argv[i], "--learning-rate"))) {
      ffconfig.learningRate = atof(argv[++i]);
      continue;
    }
    if ((!strcmp(argv[i], "--wd")) || (!strcmp(argv[i], "--weight-decay"))) {
      ffconfig.weightDecay = atof(argv[++i]);
      continue;
    }
    if ((!strcmp(argv[i], "-p")) || (!strcmp(argv[i], "--print-freq"))) {
      ffconfig.printFreq = atoi(argv[++i]);
      continue;
    }
    if ((!strcmp(argv[i], "-d")) || (!strcmp(argv[i], "--dataset"))) {
      ffconfig.dataset_path = std::string(argv[++i]);
      continue;
    }
    if ((!strcmp(argv[i], "--budget")) ||
        (!strcmp(argv[i], "--search-budget"))) {
      ffconfig.search_budget = (size_t)atoll(argv[++i]);
      continue;
    }
    if ((!strcmp(argv[i], "--alpha")) || (!strcmp(argv[i], "--search-alpha"))) {
      ffconfig.search_alpha = atof(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "--simulator-workspace-size")) {
      ffconfig.simulator_work_space_size = atoll(argv[++i]);
      continue;
    }
    if ((!strcmp(argv[i], "--import")) ||
        (!strcmp(argv[i], "--import-strategy"))) {
      ffconfig.import_strategy_file = std::string(argv[++i]);
      continue;
    }
    if ((!strcmp(argv[i], "--export")) ||
        (!strcmp(argv[i], "--export-strategy"))) {
      ffconfig.export_strategy_file = std::string(argv[++i]);
      continue;
    }
    if ((!strcmp(argv[i], "--only-data-parallel"))) {
      ffconfig.only_data_parallel = true;
      continue;
    }
    if ((!strcmp(argv[i], "--enable-parameter-parallel"))) {
      ffconfig.enable_parameter_parallel = true;
      continue;
    }
    if ((!strcmp(argv[i], "--enable-attribute-parallel"))) {
      ffconfig.enable_parameter_parallel = true;
      continue;
    }
    if (!strcmp(argv[i], "-ll:gpu")) {
      ffconfig.workersPerNode = gpus_per_node = atoi(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "--nodes")) {
      fprintf(stderr,
              "[Warning] --nodes is deprecated. "
              "FlexFlow will automatically detect the number of nodes.\n");
      ffconfig.numNodes = atoi(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "-ll:cpu")) {
      ffconfig.cpusPerNode = cpus_per_node = atoi(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "--profiling")) {
      ffconfig.profiling = true;
      continue;
    }
    if (!strcmp(argv[i], "--allow-tensor-op-math-conversion")) {
      ffconfig.allow_tensor_op_math_conversion = true;
      continue;
    }
    if (!strcmp(argv[i], "--fusion")) {
      ffconfig.perform_fusion = true;
      continue;
    }
    if (!strcmp(argv[i], "--overlap")) {
      ffconfig.search_overlap_backward_update = true;
      continue;
    }
    if (!strcmp(argv[i], "--taskgraph")) {
      ffconfig.export_strategy_task_graph_file = std::string(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "--include-costs-dot-graph")) {
      ffconfig.include_costs_dot_graph = true;
      continue;
    }
    if (!strcmp(argv[i], "--compgraph")) {
      ffconfig.export_strategy_computation_graph_file = std::string(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "--machine-model-version")) {
      ffconfig.machine_model_version = atoi(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "--machine-model-file")) {
      ffconfig.machine_model_file = std::string(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "--simulator-segment-size")) {
      ffconfig.simulator_segment_size = atoi(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "--simulator-max-num-segments")) {
      ffconfig.simulator_max_num_segments = atoi(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "--enable-propagation")) {
      ffconfig.enable_propagation = true;
      continue;
    }
    if (!strcmp(argv[i], "--enable-inplace-optimizations")) {
      ffconfig.enable_inplace_optimizations = true;
      continue;
    }
    if (!strcmp(argv[i], "--search-num-nodes")) {
      ffconfig.search_num_nodes = atoi(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "--search-num-workers")) {
      ffconfig.search_num_workers = atoi(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "--base-optimize-threshold")) {
      ffconfig.base_optimize_threshold = atoi(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "--disable-control-replication")) {
      ffconfig.enable_control_replication = false;
      continue;
    }
    if (!strcmp(argv[i], "--python-data-loader-type")) {
      ffconfig.python_data_loader_type = atoi(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "--substitution-json")) {
      ffconfig.substitution_json_path = std::string(argv[++i]);
      continue;
    }
  }
}

int main(char **argv, int argc) {
  FFConfig ffconfig;
  parse(argv, argc, ffconfig);
  return 0;
}
