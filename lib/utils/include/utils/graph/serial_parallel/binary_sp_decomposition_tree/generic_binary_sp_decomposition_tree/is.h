#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIAL_PARALLEL_BINARY_SP_DECOMPOSITION_TREE_GENERIC_BINARY_SP_DECOMPOSITION_TREE_IS_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIAL_PARALLEL_BINARY_SP_DECOMPOSITION_TREE_GENERIC_BINARY_SP_DECOMPOSITION_TREE_IS_H

#include "utils/graph/serial_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree.h"

namespace FlexFlow {

template <typename T>
bool is_series_split(GenericBinarySPDecompositionTree<T> const &t) {
  return std::holds_alternative<GenericBinarySeriesSplit<T>>(t.root);
}

template <typename T>
bool is_parallel_split(GenericBinarySPDecompositionTree<T> const &t) {
  return std::holds_alternative<GenericBinaryParallelSplit<T>>(t.root);
}

template <typename T>
bool is_leaf(GenericBinarySPDecompositionTree<T> const &t) {
  return std::holds_alternative<T>(t.root);
}

} // namespace FlexFlow

#endif
