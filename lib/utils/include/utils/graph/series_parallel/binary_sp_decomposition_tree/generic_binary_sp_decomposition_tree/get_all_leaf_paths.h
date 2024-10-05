#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIES_PARALLEL_BINARY_SP_DECOMPOSITION_TREE_GENERIC_BINARY_SP_DECOMPOSITION_TREE_GET_ALL_LEAF_PATHS_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIES_PARALLEL_BINARY_SP_DECOMPOSITION_TREE_GENERIC_BINARY_SP_DECOMPOSITION_TREE_GET_ALL_LEAF_PATHS_H

#include "utils/full_binary_tree/get_all_leaf_paths.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree.dtg.h"

namespace FlexFlow {

template <typename SeriesLabel, typename ParallelLabel, typename LeafLabel>
std::unordered_set<BinaryTreePath> get_all_leaf_paths(
    GenericBinarySPDecompositionTree<SeriesLabel,
                                     ParallelLabel,
                                     LeafLabel> const &tree) {
  return get_all_leaf_paths(tree.raw_tree);
}

} // namespace FlexFlow

#endif
