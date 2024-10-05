#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIES_PARALLEL_BINARY_SP_DECOMPOSITION_TREE_LEAF_ONLY_BINARY_SP_DECOMPOSITION_TREE_LEAF_ONLY_BINARY_SERIES_SPLIT_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIES_PARALLEL_BINARY_SP_DECOMPOSITION_TREE_LEAF_ONLY_BINARY_SP_DECOMPOSITION_TREE_LEAF_ONLY_BINARY_SERIES_SPLIT_H

#include "utils/graph/series_parallel/binary_sp_decomposition_tree/leaf_only_binary_sp_decomposition_tree/leaf_only_binary_series_split.dtg.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/leaf_only_binary_sp_decomposition_tree/leaf_only_binary_sp_decomposition_tree.dtg.h"

namespace FlexFlow {

template <typename LeafLabel>
LeafOnlyBinarySPDecompositionTree<LeafLabel>
    get_left_child(LeafOnlyBinarySeriesSplit<LeafLabel> const &s) {
  return s.pre;
}

template <typename LeafLabel>
LeafOnlyBinarySPDecompositionTree<LeafLabel>
    get_right_child(LeafOnlyBinarySeriesSplit<LeafLabel> const &s) {
  return s.post;
}

} // namespace FlexFlow

#endif
