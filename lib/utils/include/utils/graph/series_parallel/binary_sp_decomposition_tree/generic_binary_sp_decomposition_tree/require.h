#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIES_PARALLEL_BINARY_SP_DECOMPOSITION_TREE_GENERIC_BINARY_SP_DECOMPOSITION_TREE_REQUIRE_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIES_PARALLEL_BINARY_SP_DECOMPOSITION_TREE_GENERIC_BINARY_SP_DECOMPOSITION_TREE_REQUIRE_H

#include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/generic_binary_series_split.dtg.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/generic_binary_parallel_split.dtg.h"

namespace FlexFlow {

template <typename SeriesLabel, typename ParallelLabel, typename LeafLabel>
GenericBinarySeriesSplit<SeriesLabel, ParallelLabel, LeafLabel>
    require_series(GenericBinarySPDecompositionTree<SeriesLabel, ParallelLabel, LeafLabel> const &t) {
  FullBinaryTreeParentNode<std::variant<SeriesLabel, ParallelLabel>, LeafLabel> parent = require_parent_node(t.raw_tree);

  return GenericBinarySeriesSplit<SeriesLabel, ParallelLabel, LeafLabel>{
    /*label=*/std::get<SeriesLabel>(parent.label),
    /*pre=*/get_left_child(parent),
    /*post=*/get_right_child(parent),
  };
}

template <typename SeriesLabel, typename ParallelLabel, typename LeafLabel>
GenericBinaryParallelSplit<SeriesLabel, ParallelLabel, LeafLabel>
    require_parallel(GenericBinarySPDecompositionTree<SeriesLabel, ParallelLabel, LeafLabel> const &t) {
  FullBinaryTreeParentNode<std::variant<SeriesLabel, ParallelLabel>, LeafLabel> parent = require_parent_node(t.raw_tree);

  return GenericBinarySeriesSplit<SeriesLabel, ParallelLabel, LeafLabel>{
    /*label=*/std::get<ParallelLabel>(parent.label),
    /*pre=*/get_left_child(parent),
    /*post=*/get_right_child(parent),
  };
}

template <typename SeriesLabel, typename ParallelLabel, typename LeafLabel>
LeafLabel require_leaf(GenericBinarySPDecompositionTree<SeriesLabel, ParallelLabel, LeafLabel> const &t) {
  return require_leaf(t.raw_tree);
}

} // namespace FlexFlow

#endif
