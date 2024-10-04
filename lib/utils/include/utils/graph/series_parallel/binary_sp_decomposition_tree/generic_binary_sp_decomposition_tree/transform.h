#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIES_PARALLEL_BINARY_SP_DECOMPOSITION_TREE_GENERIC_BINARY_SP_DECOMPOSITION_TREE_TRANSFORM_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIES_PARALLEL_BINARY_SP_DECOMPOSITION_TREE_GENERIC_BINARY_SP_DECOMPOSITION_TREE_TRANSFORM_H

#include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/get_left_child.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/get_right_child.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/visit.h"

namespace FlexFlow {

template <typename T, typename F, typename TT = std::invoke_result_t<F, T>>
GenericBinarySPDecompositionTree<TT>
    transform(GenericBinarySPDecompositionTree<T> const &tt, F f) {
  return visit<GenericBinarySPDecompositionTree<TT>>(
      tt,
      overload{
          [&](GenericBinarySeriesSplit<T> const &s) {
            return GenericBinarySPDecompositionTree<TT>{
                GenericBinarySeriesSplit<TT>{
                    transform(get_left_child(s), f),
                    transform(get_right_child(s), f),
                },
            };
          },
          [&](GenericBinaryParallelSplit<T> const &s) {
            return GenericBinarySPDecompositionTree<TT>{
                GenericBinaryParallelSplit<TT>{
                    transform(get_left_child(s), f),
                    transform(get_right_child(s), f),
                },
            };
          },
          [&](T const &t) {
            return GenericBinarySPDecompositionTree<TT>{
                f(t),
            };
          },
      });
}

} // namespace FlexFlow

#endif
