#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIES_PARALLEL_BINARY_SP_DECOMPOSITION_TREE_GENERIC_BINARY_SP_DECOMPOSITION_TREE_GET_LEFT_CHILD_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIES_PARALLEL_BINARY_SP_DECOMPOSITION_TREE_GENERIC_BINARY_SP_DECOMPOSITION_TREE_GET_LEFT_CHILD_H

#include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree.dtg.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/generic_binary_series_split.dtg.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/generic_binary_parallel_split.dtg.h"

namespace FlexFlow {

template <typename SeriesLabel, typename ParallelLabel, typename LeafLabel>
GenericBinarySPDecompositionTree<SeriesLabel, ParallelLabel, LeafLabel>
    get_left_child(GenericBinarySeriesSplit<SeriesLabel, ParallelLabel, LeafLabel> const &s) {
  return s.pre;
}

template <typename SeriesLabel, typename ParallelLabel, typename LeafLabel>
GenericBinarySPDecompositionTree<SeriesLabel, ParallelLabel, LeafLabel>
    get_left_child(GenericBinaryParallelSplit<SeriesLabel, ParallelLabel, LeafLabel> const &p) {
  return p.lhs;
}

} // namespace FlexFlow

#endif
