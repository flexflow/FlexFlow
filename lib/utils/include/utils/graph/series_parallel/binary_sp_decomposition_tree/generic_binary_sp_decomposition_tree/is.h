#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIES_PARALLEL_BINARY_SP_DECOMPOSITION_TREE_GENERIC_BINARY_SP_DECOMPOSITION_TREE_IS_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIES_PARALLEL_BINARY_SP_DECOMPOSITION_TREE_GENERIC_BINARY_SP_DECOMPOSITION_TREE_IS_H

#include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/get_node_type.h"

namespace FlexFlow {

template <typename SeriesLabel, typename ParallelLabel, typename LeafLabel>
bool is_series_split(GenericBinarySPDecompositionTree<SeriesLabel, ParallelLabel, LeafLabel> const &t) {
  return get_node_type(t) == SPDecompositionTreeNodeType::SERIES;
}

template <typename SeriesLabel, typename ParallelLabel, typename LeafLabel>
bool is_parallel_split(GenericBinarySPDecompositionTree<SeriesLabel, ParallelLabel, LeafLabel> const &t) {
  return get_node_type(t) == SPDecompositionTreeNodeType::PARALLEL;
}

template <typename SeriesLabel, typename ParallelLabel, typename LeafLabel>
bool is_leaf(GenericBinarySPDecompositionTree<SeriesLabel, ParallelLabel, LeafLabel> const &t) {
  return get_node_type(t) == SPDecompositionTreeNodeType::NODE;
}

} // namespace FlexFlow

#endif
