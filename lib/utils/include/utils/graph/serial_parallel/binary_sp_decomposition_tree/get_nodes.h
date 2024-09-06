#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIAL_PARALLEL_BINARY_SP_DECOMPOSITION_TREE_GET_NODES_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIAL_PARALLEL_BINARY_SP_DECOMPOSITION_TREE_GET_NODES_H

#include "utils/containers/multiset_union.h"
#include "utils/graph/serial_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree.h"
#include "utils/overload.h"
#include <unordered_set>

namespace FlexFlow {

template <typename T>
std::unordered_multiset<T>
    get_nodes(GenericBinarySPDecompositionTree<T> const &tt) {
  return tt.template visit<std::unordered_multiset<T>>(overload{
      [](T const &t) { return std::unordered_multiset{t}; },
      [](GenericBinarySeriesSplit<T> const &s) { return get_nodes(s); },
      [](GenericBinaryParallelSplit<T> const &p) { return get_nodes(p); },
  });
}

template <typename T>
std::unordered_multiset<T> get_nodes(GenericBinarySeriesSplit<T> const &s) {
  return multiset_union(get_nodes(s.left_child()), get_nodes(s.right_child()));
}

template <typename T>
std::unordered_multiset<T> get_nodes(GenericBinaryParallelSplit<T> const &p) {
  return multiset_union(get_nodes(p.left_child()), get_nodes(p.right_child()));
}

} // namespace FlexFlow

#endif
