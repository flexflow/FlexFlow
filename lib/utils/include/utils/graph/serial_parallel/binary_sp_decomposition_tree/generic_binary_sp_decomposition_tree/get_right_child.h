#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIAL_PARALLEL_BINARY_SP_DECOMPOSITION_TREE_GENERIC_BINARY_SP_DECOMPOSITION_TREE_GET_RIGHT_CHILD_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIAL_PARALLEL_BINARY_SP_DECOMPOSITION_TREE_GENERIC_BINARY_SP_DECOMPOSITION_TREE_GET_RIGHT_CHILD_H

#include "utils/graph/serial_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree.h"
#include "utils/graph/serial_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/visit.h"
#include "utils/overload.h"
#include "utils/exception.h"

namespace FlexFlow {

template <typename T>
GenericBinarySPDecompositionTree<T> 
  get_right_child(GenericBinarySeriesSplit<T> const &s) {
  return *s.right_child_ptr;
}

template <typename T>
GenericBinarySPDecompositionTree<T>
  get_right_child(GenericBinaryParallelSplit<T> const &p) {
  return *p.right_child_ptr;
}

template <typename T>
GenericBinarySPDecompositionTree<T>
    get_right_child(GenericBinarySPDecompositionTree<T> const &tt) {
  return visit<GenericBinarySPDecompositionTree<T>>(tt, overload{
      [](GenericBinarySeriesSplit<T> const &s) { return get_right_child(s); },
      [](GenericBinaryParallelSplit<T> const &p) { return get_right_child(p); },
      [](T const &t) -> GenericBinarySPDecompositionTree<T> {
        throw mk_runtime_error(
            "get_right_child incorrectly called on leaf node");
      },
  });
}

} // namespace FlexFlow

#endif
