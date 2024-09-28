#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIES_PARALLEL_BINARY_SP_DECOMPOSITION_TREE_GENERIC_BINARY_SP_DECOMPOSITION_TREE_REQUIRE_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIES_PARALLEL_BINARY_SP_DECOMPOSITION_TREE_GENERIC_BINARY_SP_DECOMPOSITION_TREE_REQUIRE_H

#include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/get.h"

namespace FlexFlow {

template <typename T>
GenericBinarySeriesSplit<T> const &
    require_series(GenericBinarySPDecompositionTree<T> const &t) {
  return get<GenericBinarySeriesSplit<T>>(t);
}

template <typename T>
GenericBinaryParallelSplit<T> const &
    require_parallel(GenericBinarySPDecompositionTree<T> const &t) {
  return get<GenericBinaryParallelSplit<T>>(t);
}

template <typename T>
T const &require_leaf(GenericBinarySPDecompositionTree<T> const &t) {
  return get<T>(t);
}

} // namespace FlexFlow

#endif
