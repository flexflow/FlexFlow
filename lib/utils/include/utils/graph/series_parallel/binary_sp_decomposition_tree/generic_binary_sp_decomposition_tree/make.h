#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIES_PARALLEL_BINARY_SP_DECOMPOSITION_TREE_GENERIC_BINARY_SP_DECOMPOSITION_TREE_MAKE_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIES_PARALLEL_BINARY_SP_DECOMPOSITION_TREE_GENERIC_BINARY_SP_DECOMPOSITION_TREE_MAKE_H

#include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree.h"

namespace FlexFlow {

template <typename T>
GenericBinarySPDecompositionTree<T> make_generic_binary_series_split(
    GenericBinarySPDecompositionTree<T> const &lhs,
    GenericBinarySPDecompositionTree<T> const &rhs) {
  return GenericBinarySPDecompositionTree<T>{
      GenericBinarySeriesSplit<T>{
          lhs,
          rhs,
      },
  };
}

template <typename T>
GenericBinarySPDecompositionTree<T> make_generic_binary_parallel_split(
    GenericBinarySPDecompositionTree<T> const &lhs,
    GenericBinarySPDecompositionTree<T> const &rhs) {
  return GenericBinarySPDecompositionTree<T>{
      GenericBinaryParallelSplit<T>{
          lhs,
          rhs,
      },
  };
}

template <typename T>
GenericBinarySPDecompositionTree<T> make_generic_binary_sp_leaf(T const &t) {
  return GenericBinarySPDecompositionTree<T>{t};
}

} // namespace FlexFlow

#endif
