#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIES_PARALLEL_BINARY_SP_DECOMPOSITION_TREE_GENERIC_BINARY_SP_DECOMPOSITION_TREE_WRAP_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIES_PARALLEL_BINARY_SP_DECOMPOSITION_TREE_GENERIC_BINARY_SP_DECOMPOSITION_TREE_WRAP_H

#include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/generic_binary_parallel_split.dtg.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/generic_binary_series_split.dtg.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree.dtg.h"

namespace FlexFlow {

template <typename SeriesLabel, typename ParallelLabel, typename LeafLabel>
GenericBinarySPDecompositionTree<SeriesLabel, ParallelLabel, LeafLabel>
  wrap_series_split(GenericBinarySeriesSplit<SeriesLabel, ParallelLabel, LeafLabel> const &series_split) {
  return GenericBinarySPDecompositionTree<SeriesLabel, ParallelLabel, LeafLabel>{
    make_full_binary_tree_parent(
      /*label=*/GenericBinarySPSplitLabel<SeriesLabel, ParallelLabel>{series_split.label},
      /*lhs=*/series_split.pre.raw_tree,
      /*rhs=*/series_split.post.raw_tree),
  };
}

template <typename SeriesLabel, typename ParallelLabel, typename LeafLabel>
GenericBinarySPDecompositionTree<SeriesLabel, ParallelLabel, LeafLabel>
  wrap_parallel_split(GenericBinaryParallelSplit<SeriesLabel, ParallelLabel, LeafLabel> const &parallel_split) {
  return GenericBinarySPDecompositionTree<SeriesLabel, ParallelLabel, LeafLabel>{
    make_full_binary_tree_parent(
      /*label=*/GenericBinarySPSplitLabel<SeriesLabel, ParallelLabel>{parallel_split.label},
      /*lhs=*/parallel_split.lhs.raw_tree,
      /*rhs=*/parallel_split.rhs.raw_tree),
  };
}


} // namespace FlexFlow

#endif
