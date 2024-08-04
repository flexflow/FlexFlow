#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIAL_PARALLEL_GRAPH_GENERATION_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIAL_PARALLEL_GRAPH_GENERATION_H

#include "utils/graph/dataflow_graph/dataflow_graph.h"
#include "utils/graph/digraph/digraph.h"
#include "utils/graph/serial_parallel/serial_parallel_decomposition.dtg.h"
namespace FlexFlow {

void parallel_extend_unsafe(DataflowGraph &g, DataflowGraphView const &ext);

void serial_extend(DataflowGraph &g, DataflowGraphView const &ext);

DataflowGraph serial_composition(DataflowGraphView const &g1,
                                 DataflowGraphView const &g2);

DataflowGraph parallel_composition(DataflowGraphView const &g1,
                                   DataflowGraphView const &g2);

std::unordered_map<Node, Node> parallel_extend(DiGraph &g,
                                               DiGraphView const &ext);
std::unordered_map<Node, Node> serial_extend(DiGraph &g,
                                             DiGraphView const &ext);

DataflowGraph dataflow_graph_from_sp_decomposition(
    SerialParallelDecomposition const &sp_decomposition);

} // namespace FlexFlow

#endif
