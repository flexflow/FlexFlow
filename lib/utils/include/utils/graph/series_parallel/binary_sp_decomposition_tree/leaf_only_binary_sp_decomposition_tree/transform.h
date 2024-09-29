#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIES_PARALLEL_BINARY_SP_DECOMPOSITION_TREE_LEAF_ONLY_BINARY_SP_DECOMPOSITION_TREE_TRANSFORM_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIES_PARALLEL_BINARY_SP_DECOMPOSITION_TREE_LEAF_ONLY_BINARY_SP_DECOMPOSITION_TREE_TRANSFORM_H

#include "utils/graph/series_parallel/binary_sp_decomposition_tree/leaf_only_binary_sp_decomposition_tree/leaf_only_binary_series_split.dtg.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/leaf_only_binary_sp_decomposition_tree/leaf_only_binary_parallel_split.dtg.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/transform.h"

namespace FlexFlow {

template <typename T, typename F, typename TT = std::invoke_result_t<F, T>>
LeafOnlyBinarySeriesSplit<TT> transform(LeafOnlyBinarySeriesSplit<T> const &t, F &&f) {
  auto ff = overload {
    [&](T const &t) {
      return f(t);  
    },
    [&](auto const &x) { 
      return x;
    },
  };

  return LeafOnlyBinarySeriesSplit<TT>{
    transform(t.pre, f),
    transform(t.post, f),
  };
}

template <typename T, typename F, typename TT = std::invoke_result_t<F, T>>
LeafOnlyBinaryParallelSplit<TT> transform(LeafOnlyBinaryParallelSplit<T> const &t, F &&f) {
  auto ff = overload {
    [&](T const &t) {
      return f(t);  
    },
    [&](auto const &x) { 
      return x;
    },
  };

  return LeafOnlyBinaryParallelSplit<TT>{
    transform(t.lhs, f),
    transform(t.rhs, f),
  };
}

template <typename T, typename F, typename TT = std::invoke_result_t<F, T>>
LeafOnlyBinarySPDecompositionTree<TT> transform(LeafOnlyBinarySPDecompositionTree<T> const &t, F &&f) {
  auto ff = overload {
    [&](T const &t) {
      return f(t);  
    },
    [&](auto const &x) { 
      return x;
    },
  };

  return LeafOnlyBinarySPDecompositionTree<TT>{
    transform(t.raw_tree, ff),
  };
}


} // namespace FlexFlow

#endif
