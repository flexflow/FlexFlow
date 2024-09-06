#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIAL_PARALLEL_BINARY_SP_DECOMPOSITION_TREE_IS_BINARY_SP_TREE_RIGHT_ASSOCIATIVE_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIAL_PARALLEL_BINARY_SP_DECOMPOSITION_TREE_IS_BINARY_SP_TREE_RIGHT_ASSOCIATIVE_H

#include "utils/graph/serial_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree.h"

namespace FlexFlow {

template <typename T>
bool is_binary_sp_tree_right_associative(GenericBinarySPDecompositionTree<T> const &tt) {
  return tt.template visit<bool>(overload {
    [](T const &t) { return true; },
    [](GenericBinarySeriesSplit<T> const &s) { 
      return !s.left_child().template has<GenericBinarySeriesSplit<T>>()
        && is_binary_sp_tree_right_associative(s.left_child())
        && is_binary_sp_tree_right_associative(s.right_child());
    },
    [](GenericBinaryParallelSplit<T> const &p) {
      return !p.left_child().template has<GenericBinaryParallelSplit<T>>()
        && is_binary_sp_tree_right_associative(p.left_child())
        && is_binary_sp_tree_right_associative(p.right_child());
    },
  });
}

} // namespace FlexFlow

#endif
