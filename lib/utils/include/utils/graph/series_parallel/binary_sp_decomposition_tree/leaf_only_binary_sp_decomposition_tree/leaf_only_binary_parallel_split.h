#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIES_PARALLEL_BINARY_SP_DECOMPOSITION_TREE_LEAF_ONLY_BINARY_SP_DECOMPOSITION_TREE_LEAF_ONLY_BINARY_PARALLEL_SPLIT_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIES_PARALLEL_BINARY_SP_DECOMPOSITION_TREE_LEAF_ONLY_BINARY_SP_DECOMPOSITION_TREE_LEAF_ONLY_BINARY_PARALLEL_SPLIT_H

#include "utils/graph/series_parallel/binary_sp_decomposition_tree/leaf_only_binary_sp_decomposition_tree/leaf_only_binary_parallel_split.dtg.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/leaf_only_binary_sp_decomposition_tree/leaf_only_binary_sp_decomposition_tree.dtg.h"

namespace FlexFlow {

template <typename LeafLabel>
LeafOnlyBinarySPDecompositionTree<LeafLabel> get_left_child(LeafOnlyBinaryParallelSplit<LeafLabel> const &s) {
  return s.lhs;
}

template <typename LeafLabel>
LeafOnlyBinarySPDecompositionTree<LeafLabel> get_right_child(LeafOnlyBinaryParallelSplit<LeafLabel> const &s) {
  return s.rhs;
}

} // namespace FlexFlow

#endif
