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

bool is_empty(Node const &node);
bool is_empty(SerialSplit const &serial);
bool is_empty(ParallelSplit const &parallel);
bool is_empty(SerialParallelDecomposition const &sp);

// duplicate nodes within `sp` are counted multiple times
size_t num_nodes(SerialParallelDecomposition const &sp);

SerialParallelDecomposition serial_composition(
    std::vector<SerialParallelDecomposition> const &sp_compositions);
SerialParallelDecomposition parallel_composition(
    std::unordered_set<SerialParallelDecomposition> const &sp_compositions);

} // namespace FlexFlow

#endif
