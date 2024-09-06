#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIAL_PARALLEL_DIGRAPH_GENERATION_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIAL_PARALLEL_DIGRAPH_GENERATION_H

#include "utils/graph/digraph/digraph.h"
#include "utils/graph/serial_parallel/serial_parallel_decomposition.dtg.h"

namespace FlexFlow {

std::unordered_map<Node, Node> parallel_extend(DiGraph &g,
                                               DiGraphView const &ext);
std::unordered_map<Node, Node> serial_extend(DiGraph &g,
                                             DiGraphView const &ext);
DiGraph serial_composition(DiGraphView const &g1, DiGraphView const &g2);
DiGraph parallel_composition(DiGraphView const &g1, DiGraphView const &g2);
DiGraph serial_composition(std::vector<DiGraphView> const &graphs);
DiGraph parallel_composition(std::vector<DiGraphView> const &graphs);

DiGraph digraph_from_sp_decomposition(SerialParallelDecomposition const &sp);

} // namespace FlexFlow

#endif
