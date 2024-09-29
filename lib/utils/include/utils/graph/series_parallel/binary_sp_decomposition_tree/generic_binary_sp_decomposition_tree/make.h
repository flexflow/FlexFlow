#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIES_PARALLEL_BINARY_SP_DECOMPOSITION_TREE_GENERIC_BINARY_SP_DECOMPOSITION_TREE_MAKE_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIES_PARALLEL_BINARY_SP_DECOMPOSITION_TREE_GENERIC_BINARY_SP_DECOMPOSITION_TREE_MAKE_H

#include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree.dtg.h"

namespace FlexFlow {

template <typename SeriesLabel, typename ParallelLabel, typename LeafLabel>
GenericBinarySPDecompositionTree<SeriesLabel, ParallelLabel, LeafLabel> make_generic_binary_series_split(
    SeriesLabel const &label,
    GenericBinarySPDecompositionTree<SeriesLabel, ParallelLabel, LeafLabel> const &lhs,
    GenericBinarySPDecompositionTree<SeriesLabel, ParallelLabel, LeafLabel> const &rhs) {
  return GenericBinarySPDecompositionTree<SeriesLabel, ParallelLabel, LeafLabel>{
    FullBinaryTree<std::variant<SeriesLabel, ParallelLabel>, LeafLabel>{
      FullBinaryTreeParentNode<std::variant<SeriesLabel, ParallelLabel>, LeafLabel>{
        label, 
        lhs.raw_tree,
        rhs.raw_tree,
      }
    }
  };
}

template <typename SeriesLabel, typename ParallelLabel, typename LeafLabel>
GenericBinarySPDecompositionTree<SeriesLabel, ParallelLabel, LeafLabel> make_generic_binary_parallel_split(
    SeriesLabel const &label,
    GenericBinarySPDecompositionTree<SeriesLabel, ParallelLabel, LeafLabel> const &lhs,
    GenericBinarySPDecompositionTree<SeriesLabel, ParallelLabel, LeafLabel> const &rhs) {
  return GenericBinarySPDecompositionTree<SeriesLabel, ParallelLabel, LeafLabel>{
    FullBinaryTree<std::variant<SeriesLabel, ParallelLabel>, LeafLabel>{
      FullBinaryTreeParentNode<std::variant<SeriesLabel, ParallelLabel>, LeafLabel>{
        label, 
        lhs.raw_tree,
        rhs.raw_tree,
      }
    }
  };
}

template <typename SeriesLabel, typename ParallelLabel, typename LeafLabel>
GenericBinarySPDecompositionTree<SeriesLabel, ParallelLabel, LeafLabel> make_generic_binary_sp_leaf(LeafLabel const &leaf) {
  return GenericBinarySPDecompositionTree<SeriesLabel, ParallelLabel, LeafLabel>{
    FullBinaryTree<std::variant<SeriesLabel, ParallelLabel>, LeafLabel>{
      leaf,
    },
  };
}

} // namespace FlexFlow

#endif
