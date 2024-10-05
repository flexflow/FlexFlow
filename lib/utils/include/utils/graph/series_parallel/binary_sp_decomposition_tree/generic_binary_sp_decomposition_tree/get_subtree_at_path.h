#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIES_PARALLEL_BINARY_SP_DECOMPOSITION_TREE_GENERIC_BINARY_SP_DECOMPOSITION_TREE_GET_SUBTREE_AT_PATH_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIES_PARALLEL_BINARY_SP_DECOMPOSITION_TREE_GENERIC_BINARY_SP_DECOMPOSITION_TREE_GET_SUBTREE_AT_PATH_H

#include "utils/full_binary_tree/binary_tree_path.dtg.h"
#include "utils/full_binary_tree/get_subtree_at_path.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree.dtg.h"
#include <optional>

namespace FlexFlow {

template <typename SeriesSplitLabel,
          typename ParallelSplitLabel,
          typename LeafLabel>
std::optional<GenericBinarySPDecompositionTree<SeriesSplitLabel,
                                               ParallelSplitLabel,
                                               LeafLabel>>
    get_subtree_at_path(GenericBinarySPDecompositionTree<SeriesSplitLabel,
                                                         ParallelSplitLabel,
                                                         LeafLabel> const &tree,
                        BinaryTreePath const &path) {
  std::optional<FullBinaryTree<
      GenericBinarySPSplitLabel<SeriesSplitLabel, ParallelSplitLabel>,
      LeafLabel>>
      raw_subtree = get_subtree_at_path(tree.raw_tree, path);

  if (!raw_subtree.has_value()) {
    return std::nullopt;
  } else {
    return GenericBinarySPDecompositionTree<SeriesSplitLabel,
                                            ParallelSplitLabel,
                                            LeafLabel>{
        raw_subtree.value(),
    };
  }
}

} // namespace FlexFlow

#endif
