#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIES_PARALLEL_BINARY_SP_DECOMPOSITION_TREE_GENERIC_BINARY_SP_DECOMPOSITION_TREE_TRANSFORM_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIES_PARALLEL_BINARY_SP_DECOMPOSITION_TREE_GENERIC_BINARY_SP_DECOMPOSITION_TREE_TRANSFORM_H

#include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree.dtg.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/get_left_child.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/get_right_child.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/visit.h"
#include "utils/overload.h"

namespace FlexFlow {

template <typename SeriesLabel, 
          typename ParallelLabel,
          typename LeafLabel,
          typename F, 
          typename SeriesLabel2 = std::invoke_result_t<F, SeriesLabel>,
          typename ParallelLabel2 = std::invoke_result_t<F, ParallelLabel>,
          typename LeafLabel2 = std::invoke_result_t<F, LeafLabel>>
GenericBinarySeriesSplit<SeriesLabel2, ParallelLabel2, LeafLabel2>
    transform(GenericBinarySeriesSplit<SeriesLabel2, ParallelLabel2, LeafLabel2> const &s, F f) {
  return GenericBinarySeriesSplit<SeriesLabel2, ParallelLabel2, LeafLabel2>{
          f(s.label),
          transform(get_left_child(s), f),
          transform(get_right_child(s), f),
      };
};

template <typename SeriesLabel, 
          typename ParallelLabel,
          typename LeafLabel,
          typename F, 
          typename SeriesLabel2 = std::invoke_result_t<F, SeriesLabel>,
          typename ParallelLabel2 = std::invoke_result_t<F, ParallelLabel>,
          typename LeafLabel2 = std::invoke_result_t<F, LeafLabel>>
GenericBinaryParallelSplit<SeriesLabel2, ParallelLabel2, LeafLabel2>
    transform(GenericBinaryParallelSplit<SeriesLabel2, ParallelLabel2, LeafLabel2> const &s, F f) {
  return GenericBinaryParallelSplit<SeriesLabel2, ParallelLabel2, LeafLabel2>{
          f(s.label),
          transform(get_left_child(s), f),
          transform(get_right_child(s), f),
      };
};

template <typename SeriesLabel, 
          typename ParallelLabel,
          typename LeafLabel,
          typename F, 
          typename SeriesLabel2 = std::invoke_result_t<F, SeriesLabel>,
          typename ParallelLabel2 = std::invoke_result_t<F, ParallelLabel>,
          typename LeafLabel2 = std::invoke_result_t<F, LeafLabel>>
GenericBinarySPDecompositionTree<SeriesLabel2, ParallelLabel2, LeafLabel2>
    transform(GenericBinarySPDecompositionTree<SeriesLabel, ParallelLabel, LeafLabel> const &tt, F f) {
  return visit<GenericBinarySPDecompositionTree<SeriesLabel2, ParallelLabel2, LeafLabel2>>(
      tt,
      overload{
          [&](GenericBinarySeriesSplit<SeriesLabel, ParallelLabel, LeafLabel> const &s) {
            return GenericBinarySPDecompositionTree<SeriesLabel2, ParallelLabel2, LeafLabel2>{
              transform(s, f),
            };
          },
          [&](GenericBinaryParallelSplit<SeriesLabel, ParallelLabel, LeafLabel> const &s) {
            return GenericBinarySPDecompositionTree<SeriesLabel2, ParallelLabel2, LeafLabel2>{
              transform(s, f),
            };
          },
          [&](LeafLabel const &t) {
            return GenericBinarySPDecompositionTree<SeriesLabel2, ParallelLabel2, LeafLabel2>{
                f(t),
            };
          },
      });
}

} // namespace FlexFlow

#endif
