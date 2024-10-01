#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIES_PARALLEL_BINARY_SP_DECOMPOSITION_TREE_GENERIC_BINARY_SP_DECOMPOSITION_TREE_TRANSFORM_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIES_PARALLEL_BINARY_SP_DECOMPOSITION_TREE_GENERIC_BINARY_SP_DECOMPOSITION_TREE_TRANSFORM_H

#include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree.dtg.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree_visitor.dtg.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/get_left_child.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/get_right_child.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/visit.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/wrap.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/make.h"
#include "utils/overload.h"

namespace FlexFlow {

template <typename SeriesLabel, 
          typename ParallelLabel,
          typename LeafLabel,
          typename SeriesLabel2,
          typename ParallelLabel2,
          typename LeafLabel2>
GenericBinarySPDecompositionTree<SeriesLabel2, ParallelLabel2, LeafLabel2>
    transform(GenericBinarySPDecompositionTree<SeriesLabel, ParallelLabel, LeafLabel> const &tt, 
              GenericBinarySPDecompositionTreeVisitor<SeriesLabel, ParallelLabel, LeafLabel, SeriesLabel2, ParallelLabel2, LeafLabel2> const &visitor);

template <typename SeriesLabel, 
          typename ParallelLabel,
          typename LeafLabel,
          typename SeriesLabel2,
          typename ParallelLabel2,
          typename LeafLabel2>
GenericBinarySeriesSplit<SeriesLabel2, ParallelLabel2, LeafLabel2>
    transform(GenericBinarySeriesSplit<SeriesLabel, ParallelLabel, LeafLabel> const &s, 
              GenericBinarySPDecompositionTreeVisitor<SeriesLabel, ParallelLabel, LeafLabel, SeriesLabel2, ParallelLabel2, LeafLabel2> const &visitor) {
  return GenericBinarySeriesSplit<SeriesLabel2, ParallelLabel2, LeafLabel2>{
          visitor.series_split_func(s.label),
          transform(get_left_child(s), visitor),
          transform(get_right_child(s), visitor),
      };
};

template <typename SeriesLabel, 
          typename ParallelLabel,
          typename LeafLabel,
          typename SeriesLabel2,
          typename ParallelLabel2,
          typename LeafLabel2>
GenericBinaryParallelSplit<SeriesLabel2, ParallelLabel2, LeafLabel2>
    transform(GenericBinaryParallelSplit<SeriesLabel, ParallelLabel, LeafLabel> const &s, 
              GenericBinarySPDecompositionTreeVisitor<SeriesLabel, ParallelLabel, LeafLabel, SeriesLabel2, ParallelLabel2, LeafLabel2> const &visitor) {
  return GenericBinaryParallelSplit<SeriesLabel2, ParallelLabel2, LeafLabel2>{
          visitor.parallel_split_func(s.label),
          transform(get_left_child(s), visitor),
          transform(get_right_child(s), visitor),
      };
};

template <typename SeriesLabel, 
          typename ParallelLabel,
          typename LeafLabel,
          typename SeriesLabel2,
          typename ParallelLabel2,
          typename LeafLabel2>
GenericBinarySPDecompositionTree<SeriesLabel2, ParallelLabel2, LeafLabel2>
    transform(GenericBinarySPDecompositionTree<SeriesLabel, ParallelLabel, LeafLabel> const &tt, 
              GenericBinarySPDecompositionTreeVisitor<SeriesLabel, ParallelLabel, LeafLabel, SeriesLabel2, ParallelLabel2, LeafLabel2> const &visitor) {
  return visit<GenericBinarySPDecompositionTree<SeriesLabel2, ParallelLabel2, LeafLabel2>>(
      tt,
      overload{
          [&](GenericBinarySeriesSplit<SeriesLabel, ParallelLabel, LeafLabel> const &s) {
            return wrap_series_split(transform(s, visitor));
          },
          [&](GenericBinaryParallelSplit<SeriesLabel, ParallelLabel, LeafLabel> const &s) {
            return wrap_parallel_split(transform(s, visitor));
          },
          [&](LeafLabel const &t) {
            return make_generic_binary_sp_leaf<SeriesLabel2, ParallelLabel2>(visitor.leaf_func(t));
          },
      });
}

} // namespace FlexFlow

#endif
