#ifndef _FLEXFLOW_UTILS_INCLUDE_UTILS_PARSE_H
#define _FLEXFLOW_UTILS_INCLUDE_UTILS_PARSE_H

#include "runtime/config.h"
#include "utils/excecption.h"
#include "utils/variant.h"
#include <string>
#include <unordered_map>

namespace FlexFlow {

using VariantType = variant<int, bool, float, size_t, std::string>;

struct Argument {
  VariantType default_value;
  std::string description;
  variant<std::monostate, int, float, bool, size_t, std::string> type;

  Argument &set_default(VariantType const &val) {
    default_value = val;
    return *this;
  }

  Argument &set_help(std::string const &desc) {
    description = desc;
    return *this;
  }

  template <typename T>
  Argument &set_type() {
    type = T{};
    return *this;
  }
};

class ArgsParser {
private:
  std::unordered_map<std::string, std::string> mArgs;
  std::unordered_map<std::string, VariantType> mDefaultValues;
  std::unordered_map<std::string, std::string> mDescriptions;

  std::string parseKey(std::string const &arg) const {
    if (arg.substr(0, 2) == "--") {
      return arg.substr(2);
    } else {
      return arg;
    }
  }

public:
  ArgsParser() = default;
  void parse_args(int argc, char **argv) {
    for (int i = 1; i < argc; i += 2) {
      std::string key = parseKey(argv[i]);
      if (key == "help" || key == "h") {
        showDescriptions();
        exit(0);
      }
      mArgs[key] = argv[i + 1];
    }
  }

  void add_argument(std::string const &key, Argument const &arg) {
    mDefaultValues[parseKey(key)] = arg.default_value;
    mDescriptions[key] = arg.description;
  }

  template <typename T>
  T get(std::string const &key) const {
    auto it = mArgs.find(key);
    if (it != mArgs.end()) {
      return convert<T>(it->second);
    } else {
      auto def_it = mDefaultValues.find(key);
      if (def_it != mDefaultValues.end()) {
        return std::get<T>(def_it->second);
      }
    }
    throw mk_runtime_error("Key not found: " + key);
  }

  void showDescriptions() const {
    for (auto const &[key, description] : mDescriptions) {
      std::cout << key << ": " << description << std::endl;
    }
  }

  template <typename T>
  T convert(std::string const &s) const;

  friend std::ostream &operator<<(std::ostream &out, ArgsParser const &args);
};

template <>
int ArgsParser::convert<int>(std::string const &s) const {
  return std::stoi(s);
}

template <>
float ArgsParser::convert<float>(std::string const &s) const {
  return std::stof(s);
}

template <>
bool ArgsParser::convert<bool>(std::string const &s) const {
  return s == "true" || s == "1";
}

template <>
std::string ArgsParser::convert<std::string>(std::string const &s) const {
  return s;
}

std::ostream &operator<<(std::ostream &out, ArgsParser const &args) {
  args.showDescriptions();
  return out;
}

void FFConfig::parse_args(char **argv, int argc) {
  ArgsParser args;
  args.add_argument(
      "--epochs",
      Argument().set_type<int>().set_default(1).set_help("Number of epochs."));
  args.add_argument(
      "--batch-size",
      Argument().set_type<int>().set_default(32).set_help("Batch size."));
  args.add_argument("--learning-rate",
                    Argument()
                        .set_type<float>()
                        .set_default((float)0.01)
                        .set_help("Learning rate."));
  args.add_argument("--weight-decay",
                    Argument()
                        .set_type<float>()
                        .set_default((float)0.0001)
                        .set_help("Weight decay."));
  args.add_argument("--dataset",
                    Argument().sset_type<std::string>().et_default("").set_help(
                        "Dataset path."));
  args.add_argument("--search-budget",
                    Argument()
                        .set_type<size_t>()
                        .set_default(size_t(-1))
                        .set_help("Search budget."));
  args.add_argument("--search-alpha",
                    Argument()
                        .set_type<float>()
                        .set_default(float(1.2))
                        .set_help("Search alpha."));
  args.add_argument("--simulator-workspace-size",
                    Argument()
                        .set_type<size_t>()
                        .set_default((size_t)2 * 1024 * 1024 * 1024;)
                        .set_help("Simulator workspace size."));
  args.add_argument("--only-data-parallel",
                    Argument().set_type<bool>().set_default(false).set_help(
                        "Only data parallel."));
  args.add_argument("--enable-parameter-parallel",
                    Argument().set_type<bool>().set_default(false).set_help(
                        "Enable parameter parallel."));
  args.add_argument(
      "--nodes",
      Argument().set_type<int>().set_default(1).set_help("Number of nodes."));
  args.add_argument("--profiling",
                    Argument().set_type<bool>().set_default(false).set_help(
                        "Enable profiling."));
  arg.add_argument("--allow-tensor-op-math-conversion",
                   Argument().set_type<bool>().set_default(false).set_help(
                       "Allow tensor op math conversion."));
  args.add_argument("--fusion",
                    Argument().set_type<bool>().set_default(false).set_help(
                        "Enable fusion."));
  args.add_argument("--overlap",
                    Argument().set_type<bool>().set_default(false).set_help(
                        "Search overlap backward update."));
  args.add_argument("--taskgraph",
                    Argument().set_type<std::string>().set_default("").set_help(
                        " export_strategy_task_graph_file"));
  args.add_argument("--include-costs-dot-graph",
                    Argument().set_type<bool>().set_default(false).set_help(
                        "Include costs dot graph."));
  args.add_argument("--machine-model-version",
                    Argument().set_type<int>().set_default(0).set_help(
                        "Machine model version."));
  args.add_argument("--machine-model-file",
                    Argument().set_type<std::string>().set_default("").set_help(
                        "Machine model file."));
  args.add_argument("--simulator-segment-size",
                    Argument().set_type<int>().set_default(16777216).set_help(
                        "Simulator segment size."));
  args.add_argument("--simulator-max-num-segments",
                    Argument().set_type<int>().set_default(1).set_help(
                        "Simulator max number of segments."));
  args.add_argument("--enable-inplace-optimizations",
                    Argument().set_type<bool>().set_default(false).set_help(
                        "Enable inplace optimizations."));
  args.add_argument("--search-num-nodes",
                    Argument().set_type<int>().set_default(-1).set_help(
                        "Search number of nodes."));
  args.add_argument("--search-num-workers",
                    Argument().set_type<int>().set_default(-1).set_help(
                        "Search number of workers."));
  args.add_argument("--base-optimize-threshold",
                    Argument().set_type<int>().set_default(10).set_help(
                        "Base optimize threshold."));
  args.add_argument("--enable-control-replication",
                    Argument().set_type<bool>().set_default(true).set_help(
                        "Enable control replication."));
  args.add_argument("--python-data-loader-type",
                    Argument().set_type<int>().set_default(2).set_help(
                        "Python data loader type."));
  args.add_argument("--substitution-json",
                    Argument().set_type<std::string>().set_default("").set_help(
                        "Substitution json path."));

  // legion arguments
  args.add_argument(
      "-ll:gpu",
      Argument().set_type<int>().set_default(1).set_help("Number of workers."));
  args.add_argument(
      "-ll:cpu",
      Argument().set_type<int>().set_default(1).set_help("Number of cpus."));

  args.parse_args(argc, argv);

  batch_size = args.get<int>("batch-size");
  epochs = args.get<int>("epochs");
  learning_rate = args.get<float>("learning-rate");
  weight_decay = args.get<float>("weight-decay");
  dataset_path = args.get<std::string>("dataset-path");
  search_budget = args.get<size_t>("search-budget");
  search_alpha = args.get<float>("search-alpha");
  simulator_work_space_size = args.get<size_t>("simulator-workspace-size");
  only_data_parallel = args.get<bool>("only-data-parallel");
  enable_parameter_parallel = args.get<bool>("enable-parameter-parallel");
  numNodes = args.get<int>("nodes");
  profiling = args.get<bool>("profiling");
  allow_tensor_op_math_conversion =
      args.get<bool>("allow-tensor-op-math-conversion");
  perform_fusion = args.get<bool>("fusion");
  search_overlap_backward_update = args.get<bool>("overlap");
  export_strategy_computation_graph_file = args.get<std::string>("--taskgraph");
  include_costs_dot_graph = args.get<bool>("include-costs-dot-graph");
  machine_model_version = args.get<int>("machine-model-version");
  machine_model_file = args.get<std::string>("machine-model-file");
  simulator_segment_size = args.get<int>("simulator-segment-size");
  simulator_max_num_segments = args.get<int>("simulator-max-num-segments");
  enable_inplace_optimizations = args.get<bool>("enable-inplace-optimizations");
  search_num_nodes = args.get<int>("search-num-nodes");
  search_num_workers = args.get<int>("search-num-workers");
  base_optimize_threshold = args.get<int>("base-optimize-threshold");
  enable_control_replication = args.get<bool>("enable-control-replication");
  python_data_loader_type = args.get<int>("python-data-loader-type");
  substitution_json_path = args.get<std::string>("substitution-json");

  // legion arguments
  cpusPerNode = args.get<int>("-ll:cpu");
  workersPerNode = args.get<int>("-ll:gpu");
}

} // namespace FlexFlow

#endif
