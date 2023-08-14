#ifndef _FLEXFLOW_UTILS_INCLUDE_UTILS_PARSE_H
#define _FLEXFLOW_UTILS_INCLUDE_UTILS_PARSE_H

#include <argparse/argparse.hpp>
#include "runtime/config.h"

namespace FlexFlow {

void FFConfig::parse_args(char **argv, int argc) {
    argparse::ArgumentParser args("FFConfig::parse_args");
    args.add_argument("--epochs").default_value(1).help("Number of epochs.");
    args.add_argument("--batch-size").default_value(32).help("Batch size.");
    args.add_argument("--nodes").default_value(1).help("Number of nodes.");
    args.add_argument("--ll:cpu").default_value(1).help("Number of cpus.");
    args.add_argument("--ll:gpu").default_value(1).help("Number of workers.");
    args.add_argument("--learning-rate").default_value((float)0.01).help("Learning rate.");
    args.add_argument("--weight-decay").default_value((float)0.0001).help("Weight decay.");
    args.add_argument("--workSpaceSize").default_value((size_t)1 * 1024 * 1024 * 1024;).help("Workspace size.");
    args.add_argument("--profiling").default_value(false).help("Enable profiling.");
    args.add_argument("--fusion").default_value(false).help("Enable fusion.");
    args.add_argument("--simulator_work_space_size").default_value((size_t)2 * 1024 * 1024 * 1024;).help("Simulator workspace size.");
    args.add_argument("--search_budget").default_value(size_t(-1)).help("Search budget.");
    args.add_argument("--search_alpha").default_value(float(1.2)).help("Search alpha.");
    args.add_argument("--overlap").default_value(false).help("Search overlap backward update.");
    args.add_argument("--dataset_path").default_value("").help("Dataset path.");
    args.add_argument("--only_data_parallel").default_value(false).help("Only data parallel.");
    args.add_argument("--enable_parameter_parallel").default_value(false).help("Enable parameter parallel.");
    args.add_argument("--taskgraph").default_value("").help(" export_strategy_task_graph_file");
    args.add_argument("--include_costs_dot_graph").default_value(false).help("Include costs dot graph.");
    args.add_argument("--machine_model_version").default_value(0).help("Machine model version.");
    args.add_argument("--machine_model_file").default_value("").help("Machine model file.");
    args.add_argument("--simulator_segment_size").default_value(16777216).help("Simulator segment size.");
    args.add_argument("--simulator_max_num_segments").default_value(1).help("Simulator max number of segments.");
    args.add_argument("--enable_inplace_optimizations").default_value(false).help("Enable inplace optimizations.");
    args.add_argument("--search_num_nodes").default_value(-1).help("Search number of nodes.");
    args.add_argument("--search_num_workers").default_value(-1).help("Search number of workers.");
    args.add_argument("--base_optimize_threshold").default_value(10).help("Base optimize threshold.");
    args.add_argument("--enable_control_replication").default_value(true).help("Enable control replication.");
    args.add_argument("--python_data_loader_type").default_value(2).help("Python data loader type.");
    args.add_argument("--enable_control_replication").default_value(true).help("Enable control replication.");
    args.add_argument("--substitution_json_path").default_value("").help("Substitution json path.");
    arg.add_argument("--allow_tensor_op_math_conversion").default_value(false).help("Allow tensor op math conversion.");
    args.add_argument("--include_costs_dot_graph").default_value(false).help("Include costs dot graph.");
    args.parse_args(argc, argv);
    std::cout<<"args:"<<args<<std::endl;
}


} // namespace FlexFlow


#endif 
