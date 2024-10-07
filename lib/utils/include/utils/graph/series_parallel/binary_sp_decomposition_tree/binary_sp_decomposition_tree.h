#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIES_PARALLEL_BINARY_SP_DECOMPOSITION_TREE_BINARY_SP_DECOMPOSITION_TREE_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIES_PARALLEL_BINARY_SP_DECOMPOSITION_TREE_BINARY_SP_DECOMPOSITION_TREE_H

#include "utils/graph/series_parallel/binary_sp_decomposition_tree/binary_parallel_split.dtg.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/binary_series_split.dtg.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/binary_sp_decomposition_tree.dtg.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree_implementation.dtg.h"
#include "utils/graph/series_parallel/sp_decomposition_tree_node_type.dtg.h"
#include <unordered_set>

namespace FlexFlow {

GenericBinarySPDecompositionTreeImplementation<
  BinarySPDecompositionTree,
  BinarySeriesSplit,
  BinaryParallelSplit,
  Node> generic_impl_for_binary_sp_tree();

bool is_binary_sp_tree_left_associative(BinarySPDecompositionTree const &);
bool is_binary_sp_tree_right_associative(BinarySPDecompositionTree const &);

std::unordered_multiset<Node> get_leaves(BinarySPDecompositionTree const &);

SPDecompositionTreeNodeType get_node_type(BinarySPDecompositionTree const &);

} // namespace FlexFlow

#endif
