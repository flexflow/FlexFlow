#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIAL_PARALLEL_BINARY_SP_DECOMPOSITION_TREE_GET_NUM_TREE_NODES_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIAL_PARALLEL_BINARY_SP_DECOMPOSITION_TREE_GET_NUM_TREE_NODES_H

#include "utils/graph/serial_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree.h"
#include "utils/overload.h"

namespace FlexFlow {

template <typename T>
int get_num_tree_nodes(GenericBinarySPDecompositionTree<T> const &t) {
  return t.template visit<int>(overload {
    [](Node const &n) { return 1; },
    [](GenericBinarySeriesSplit<T> const &s) { return get_num_tree_nodes(s); },
    [](GenericBinaryParallelSplit<T> const &p) { return get_num_tree_nodes(p); },
  });
}

template <typename T>
int get_num_tree_nodes(GenericBinarySeriesSplit<T> const &s) {
  return 1 + get_num_tree_nodes(s.left_child()) + get_num_tree_nodes(s.right_child());
}

template <typename T>
int get_num_tree_nodes(GenericBinaryParallelSplit<T> const &p) {
  return 1 + get_num_tree_nodes(p.left_child()) + get_num_tree_nodes(p.right_child());
}

} // namespace FlexFlow

#endif
