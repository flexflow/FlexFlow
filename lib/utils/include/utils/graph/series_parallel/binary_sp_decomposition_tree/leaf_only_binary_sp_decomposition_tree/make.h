#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIES_PARALLEL_BINARY_SP_DECOMPOSITION_TREE_LEAF_ONLY_BINARY_SP_DECOMPOSITION_TREE_MAKE_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIES_PARALLEL_BINARY_SP_DECOMPOSITION_TREE_LEAF_ONLY_BINARY_SP_DECOMPOSITION_TREE_MAKE_H

#include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/make.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/leaf_only_binary_sp_decomposition_tree/leaf_only_binary_sp_decomposition_tree.dtg.h"

namespace FlexFlow {

template <typename LeafLabel>
LeafOnlyBinarySPDecompositionTree<LeafLabel> leaf_only_binary_sp_tree_make_series_split(LeafOnlyBinarySPDecompositionTree<LeafLabel> const &pre,
                                                               LeafOnlyBinarySPDecompositionTree<LeafLabel> const &post) {
  return LeafOnlyBinarySPDecompositionTree<LeafLabel>{
    make_generic_binary_series_split(
      std::monostate{},
      pre.raw_tree,
      post.raw_tree),
  };
}

template <typename LeafLabel>
LeafOnlyBinarySPDecompositionTree<LeafLabel> leaf_only_binary_sp_tree_make_parallel_split(LeafOnlyBinarySPDecompositionTree<LeafLabel> const &lhs,
                                                               LeafOnlyBinarySPDecompositionTree<LeafLabel> const &rhs) {
  return LeafOnlyBinarySPDecompositionTree<LeafLabel>{
    make_generic_binary_parallel_split(
      std::monostate{},
      lhs.raw_tree,
      rhs.raw_tree),
  };
}

template <typename LeafLabel>
LeafOnlyBinarySPDecompositionTree<LeafLabel> leaf_only_binary_sp_tree_make_leaf(LeafLabel const &label) {
  return LeafOnlyBinarySPDecompositionTree<LeafLabel>{
    make_generic_binary_sp_leaf<
      std::monostate,
      std::monostate, LeafLabel>(label),
  };
}


} // namespace FlexFlow

#endif
