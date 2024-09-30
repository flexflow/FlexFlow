#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIES_PARALLEL_BINARY_SP_DECOMPOSITION_TREE_GENERIC_BINARY_SP_DECOMPOSITION_TREE_FIND_PATHS_TO_LEAF_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIES_PARALLEL_BINARY_SP_DECOMPOSITION_TREE_GENERIC_BINARY_SP_DECOMPOSITION_TREE_FIND_PATHS_TO_LEAF_H

#include "utils/full_binary_tree/find_paths_to_leaf.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree.dtg.h"

namespace FlexFlow {

template <typename SeriesLabel, typename ParallelLabel, typename LeafLabel>
std::unordered_set<BinaryTreePath> find_paths_to_leaf(GenericBinarySPDecompositionTree<SeriesLabel, ParallelLabel, LeafLabel> const &tree,
                                                      LeafLabel const &leaf) {
  return find_paths_to_leaf(tree.raw_tree, leaf);
}

} // namespace FlexFlow

#endif
