#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIAL_PARALLEL_BINARY_SP_DECOMPOSITION_TREE_GENERIC_BINARY_SP_DECOMPOSITION_TREE_HASH_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIAL_PARALLEL_BINARY_SP_DECOMPOSITION_TREE_GENERIC_BINARY_SP_DECOMPOSITION_TREE_HASH_H

#include "utils/graph/serial_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree.h"
#include "utils/hash-utils.h"
#include "utils/hash/tuple.h"

namespace std {

template <typename T>
struct hash<::FlexFlow::GenericBinarySeriesSplit<T>> {
  size_t operator()(::FlexFlow::GenericBinarySeriesSplit<T> const &s) const {
    return get_std_hash(s.tie());
  }
};

template <typename T>
struct hash<::FlexFlow::GenericBinaryParallelSplit<T>> {
  size_t operator()(::FlexFlow::GenericBinaryParallelSplit<T> const &s) const {
    return get_std_hash(s.tie());
  }
};

template <typename T>
struct hash<::FlexFlow::GenericBinarySPDecompositionTree<T>> {
  size_t operator()(
      ::FlexFlow::GenericBinarySPDecompositionTree<T> const &s) const {
    return get_std_hash(s.tie());
  }
};

} // namespace std

#endif
