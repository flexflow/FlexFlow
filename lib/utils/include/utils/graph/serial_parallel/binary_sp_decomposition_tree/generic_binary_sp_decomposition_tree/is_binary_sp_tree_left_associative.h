#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIAL_PARALLEL_BINARY_SP_DECOMPOSITION_TREE_GENERIC_BINARY_SP_DECOMPOSITION_TREE_IS_BINARY_SP_TREE_LEFT_ASSOCIATIVE_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIAL_PARALLEL_BINARY_SP_DECOMPOSITION_TREE_GENERIC_BINARY_SP_DECOMPOSITION_TREE_IS_BINARY_SP_TREE_LEFT_ASSOCIATIVE_H

#include "utils/graph/serial_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree.h"
#include "utils/graph/serial_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/get_left_child.h"
#include "utils/graph/serial_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/get_right_child.h"
#include "utils/graph/serial_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/is.h"
#include "utils/graph/serial_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/visit.h"

namespace FlexFlow {

template <typename T>
bool is_binary_sp_tree_left_associative(
    GenericBinarySPDecompositionTree<T> const &tt) {
  return visit<bool>(
      tt,
      overload{
          [](T const &) { return true; },
          [](GenericBinarySeriesSplit<T> const &s) {
            return !is_series_split(get_right_child(s)) &&
                   is_binary_sp_tree_left_associative(get_left_child(s)) &&
                   is_binary_sp_tree_left_associative(get_right_child(s));
          },
          [](GenericBinaryParallelSplit<T> const &p) {
            return !is_parallel_split(get_right_child(p)) &&
                   is_binary_sp_tree_left_associative(get_left_child(p)) &&
                   is_binary_sp_tree_left_associative(get_right_child(p));
          },
      });
}

} // namespace FlexFlow

#endif
