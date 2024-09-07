#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIAL_PARALLEL_BINARY_SP_DECOMPOSITION_TREE_GENERIC_BINARY_SP_DECOMPOSITION_TREE_GET_LEFT_CHILD_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIAL_PARALLEL_BINARY_SP_DECOMPOSITION_TREE_GENERIC_BINARY_SP_DECOMPOSITION_TREE_GET_LEFT_CHILD_H

#include "utils/graph/serial_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree.h"
#include "utils/graph/serial_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/visit.h"
#include "utils/overload.h"
#include "utils/exception.h"

namespace FlexFlow {

template <typename T>
GenericBinarySPDecompositionTree<T> 
  get_left_child(GenericBinarySeriesSplit<T> const &s) {
  return *s.left_child_ptr;
}

template <typename T>
GenericBinarySPDecompositionTree<T>
  get_left_child(GenericBinaryParallelSplit<T> const &p) {
  return *p.left_child_ptr;
}

template <typename T>
GenericBinarySPDecompositionTree<T>
    get_left_child(GenericBinarySPDecompositionTree<T> const &tt) {
  return visit<GenericBinarySPDecompositionTree<T>>(tt, overload{
      [](GenericBinarySeriesSplit<T> const &s) { return get_left_child(s); },
      [](GenericBinaryParallelSplit<T> const &p) { return get_left_child(p); },
      [](T const &t) -> GenericBinarySPDecompositionTree<T> {
        throw mk_runtime_error(
            "get_left_child incorrectly called on leaf node");
      },
  });
}

} // namespace FlexFlow

#endif
