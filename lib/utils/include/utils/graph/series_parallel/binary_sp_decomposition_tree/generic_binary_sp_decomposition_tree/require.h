#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIES_PARALLEL_BINARY_SP_DECOMPOSITION_TREE_GENERIC_BINARY_SP_DECOMPOSITION_TREE_REQUIRE_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIES_PARALLEL_BINARY_SP_DECOMPOSITION_TREE_GENERIC_BINARY_SP_DECOMPOSITION_TREE_REQUIRE_H

#include "utils/full_binary_tree/get_label.h"
#include "utils/full_binary_tree/require.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/generic_binary_parallel_split.dtg.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/generic_binary_series_split.dtg.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/generic_binary_sp_split_label.h"

namespace FlexFlow {

template <typename SeriesLabel, typename ParallelLabel, typename LeafLabel>
GenericBinarySeriesSplit<SeriesLabel, ParallelLabel, LeafLabel>
    require_generic_binary_series_split(
        GenericBinarySPDecompositionTree<SeriesLabel,
                                         ParallelLabel,
                                         LeafLabel> const &t) {
  FullBinaryTreeParentNode<
      GenericBinarySPSplitLabel<SeriesLabel, ParallelLabel>,
      LeafLabel>
      parent = require_full_binary_tree_parent_node(t.raw_tree);

  return GenericBinarySeriesSplit<SeriesLabel, ParallelLabel, LeafLabel>{
      /*label=*/require_generic_binary_series_split_label(
          get_full_binary_tree_parent_label(parent)),
      /*pre=*/
      GenericBinarySPDecompositionTree<SeriesLabel, ParallelLabel, LeafLabel>{
          get_left_child(parent),
      },
      /*post=*/
      GenericBinarySPDecompositionTree<SeriesLabel, ParallelLabel, LeafLabel>{
          get_right_child(parent),
      },
  };
}

template <typename SeriesLabel, typename ParallelLabel, typename LeafLabel>
GenericBinaryParallelSplit<SeriesLabel, ParallelLabel, LeafLabel>
    require_generic_binary_parallel_split(
        GenericBinarySPDecompositionTree<SeriesLabel,
                                         ParallelLabel,
                                         LeafLabel> const &t) {
  FullBinaryTreeParentNode<
      GenericBinarySPSplitLabel<SeriesLabel, ParallelLabel>,
      LeafLabel>
      parent = require_full_binary_tree_parent_node(t.raw_tree);

  return GenericBinaryParallelSplit<SeriesLabel, ParallelLabel, LeafLabel>{
      /*label=*/require_generic_binary_parallel_split_label(
          get_full_binary_tree_parent_label(parent)),
      /*lhs=*/
      GenericBinarySPDecompositionTree<SeriesLabel, ParallelLabel, LeafLabel>{
          get_left_child(parent),
      },
      /*rhs=*/
      GenericBinarySPDecompositionTree<SeriesLabel, ParallelLabel, LeafLabel>{
          get_right_child(parent),
      },
  };
}

template <typename SeriesLabel, typename ParallelLabel, typename LeafLabel>
LeafLabel require_generic_binary_leaf(
    GenericBinarySPDecompositionTree<SeriesLabel,
                                     ParallelLabel,
                                     LeafLabel> const &t) {
  return require_full_binary_tree_leaf(t.raw_tree);
}

} // namespace FlexFlow

#endif
