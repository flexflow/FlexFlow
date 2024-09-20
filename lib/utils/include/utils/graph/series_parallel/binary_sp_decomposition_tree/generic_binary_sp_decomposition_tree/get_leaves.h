#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIES_PARALLEL_BINARY_SP_DECOMPOSITION_TREE_GENERIC_BINARY_SP_DECOMPOSITION_TREE_GET_LEAVES_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIES_PARALLEL_BINARY_SP_DECOMPOSITION_TREE_GENERIC_BINARY_SP_DECOMPOSITION_TREE_GET_LEAVES_H

#include "utils/containers/multiset_union.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/get_left_child.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/get_right_child.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/visit.h"
#include "utils/overload.h"
#include <unordered_set>

namespace FlexFlow {

template <typename T>
std::unordered_multiset<T>
    get_leaves(GenericBinarySPDecompositionTree<T> const &tt) {
  return visit<std::unordered_multiset<T>>(
      tt,
      overload{
          [](T const &t) { return std::unordered_multiset{t}; },
          [](GenericBinarySeriesSplit<T> const &s) { return get_leaves(s); },
          [](GenericBinaryParallelSplit<T> const &p) { return get_leaves(p); },
      });
}

template <typename T>
std::unordered_multiset<T> get_leaves(GenericBinarySeriesSplit<T> const &s) {
  return multiset_union(get_leaves(get_left_child(s)),
                        get_leaves(get_right_child(s)));
}

template <typename T>
std::unordered_multiset<T> get_leaves(GenericBinaryParallelSplit<T> const &p) {
  return multiset_union(get_leaves(get_left_child(p)),
                        get_leaves(get_right_child(p)));
}

} // namespace FlexFlow

#endif
