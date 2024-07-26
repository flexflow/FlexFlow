#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIAL_PARALLEL_SERIAL_PARALLEL_DECOMPOSITION_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIAL_PARALLEL_SERIAL_PARALLEL_DECOMPOSITION_H

#include "utils/graph/serial_parallel/intermediate_sp_decomposition_tree.dtg.h"
#include "utils/graph/serial_parallel/serial_parallel_decomposition.dtg.h"
#include <variant>

namespace FlexFlow {

std::variant<SerialSplit, ParallelSplit, Node> internal_to_final_ast(
    std::variant<IntermediateSpDecompositionTree, Node> const &ast);
SerialParallelDecomposition
    to_final_ast(std::variant<IntermediateSpDecompositionTree, Node> const &);

std::unordered_set<Node> get_nodes(SerialParallelDecomposition const &sp);
std::unordered_set<Node> get_nodes(SerialSplit const &);
std::unordered_set<Node> get_nodes(ParallelSplit const &);
std::unordered_set<Node> get_nodes(Node const &);

} // namespace FlexFlow

#endif
