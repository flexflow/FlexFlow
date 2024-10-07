#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIES_PARALLEL_BINARY_SP_DECOMPOSITION_TREE_GENERIC_BINARY_SP_DECOMPOSITION_TREE_REQUIRE_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIES_PARALLEL_BINARY_SP_DECOMPOSITION_TREE_GENERIC_BINARY_SP_DECOMPOSITION_TREE_REQUIRE_H

#include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/generic_binary_parallel_split.dtg.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/generic_binary_series_split.dtg.h"

namespace FlexFlow {

template <typename SeriesLabel, typename ParallelLabel, typename LeafLabel>
GenericBinarySeriesSplit<SeriesLabel, ParallelLabel, LeafLabel>
    require_generic_binary_series_split(
        GenericBinarySPDecompositionTree<SeriesLabel,
                                         ParallelLabel,
                                         LeafLabel> const &tree) {
  return tree.template get<GenericBinarySeriesSplit<SeriesLabel, ParallelLabel, LeafLabel>>();
}

template <typename SeriesLabel, typename ParallelLabel, typename LeafLabel>
GenericBinaryParallelSplit<SeriesLabel, ParallelLabel, LeafLabel>
    require_generic_binary_parallel_split(
        GenericBinarySPDecompositionTree<SeriesLabel,
                                         ParallelLabel,
                                         LeafLabel> const &tree) {
  return tree.template get<GenericBinaryParallelSplit<SeriesLabel, ParallelLabel, LeafLabel>>();
}

template <typename SeriesLabel, typename ParallelLabel, typename LeafLabel>
LeafLabel require_generic_binary_leaf(
    GenericBinarySPDecompositionTree<SeriesLabel,
                                     ParallelLabel,
                                     LeafLabel> const &tree) {
  return tree.template get<LeafLabel>();
}

} // namespace FlexFlow

#endif
