#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIAL_PARALLEL_BINARY_SP_DECOMPOSITION_TREE_GET_NODES_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIAL_PARALLEL_BINARY_SP_DECOMPOSITION_TREE_GET_NODES_H

#include "utils/graph/node/node.dtg.h"
#include "utils/graph/serial_parallel/binary_sp_decomposition_tree/binary_sp_decomposition_tree.h"
#include <unordered_set>

namespace FlexFlow {

std::unordered_multiset<Node> get_nodes(BinarySPDecompositionTree const &);
std::unordered_multiset<Node> get_nodes(BinarySeriesSplit const &);
std::unordered_multiset<Node> get_nodes(BinaryParallelSplit const &);

} // namespace FlexFlow

#endif
