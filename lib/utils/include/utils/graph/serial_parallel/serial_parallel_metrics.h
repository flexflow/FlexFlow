#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIAL_PARALLEL_METRICS_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIAL_PARALLEL_METRICS_H

#include "utils/graph/digraph/digraph_view.h"
#include "utils/graph/serial_parallel/serial_parallel_decomposition.dtg.h"
#include <unordered_map>

namespace FlexFlow {

std::unordered_map<Node, size_t> get_node_frequency_map(Node const &node);
std::unordered_map<Node, size_t>
    get_node_frequency_map(SerialSplit const &serial);
std::unordered_map<Node, size_t>
    get_node_frequency_map(ParallelSplit const &parallel);
std::unordered_map<Node, size_t>
    get_node_frequency_map(SerialParallelDecomposition const &sp);

float work_cost(SerialParallelDecomposition const &sp,
                std::unordered_map<Node, float> cost_map);

float work_cost(DiGraphView const &g,
                std::unordered_map<Node, float> const &cost_map);

int num_dependencies(SerialParallelDecomposition const &sp);

int num_dependencies(DiGraphView const &g);

float critical_path_cost(SerialParallelDecomposition const &sp,
                         std::unordered_map<Node, float> const &cost_map);

float critical_path_cost(DiGraphView const &g,
                         std::unordered_map<Node, float> const &cost_map);

float relative_work_increase(DiGraphView const &g,
                             SerialParallelDecomposition const &sp,
                             std::unordered_map<Node, float> const &cost_map);

float relative_critical_path_cost_increase(
    DiGraphView const &g,
    SerialParallelDecomposition const &sp,
    std::unordered_map<Node, float> const &cost_map);

float relative_num_dependencies_increase(DiGraphView const &g,
                                         SerialParallelDecomposition const &sp);

} // namespace FlexFlow

#endif // FLEXFLOW_SERIAL_PARALLEL_METRICS_H
