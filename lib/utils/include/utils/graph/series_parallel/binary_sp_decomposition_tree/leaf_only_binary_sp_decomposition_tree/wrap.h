#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIES_PARALLEL_BINARY_SP_DECOMPOSITION_TREE_LEAF_ONLY_BINARY_SP_DECOMPOSITION_TREE_WRAP_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIES_PARALLEL_BINARY_SP_DECOMPOSITION_TREE_LEAF_ONLY_BINARY_SP_DECOMPOSITION_TREE_WRAP_H

#include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/generic_binary_series_split.dtg.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/wrap.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/leaf_only_binary_sp_decomposition_tree/leaf_only_binary_parallel_split.dtg.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/leaf_only_binary_sp_decomposition_tree/leaf_only_binary_series_split.dtg.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/leaf_only_binary_sp_decomposition_tree/leaf_only_binary_sp_decomposition_tree.dtg.h"

namespace FlexFlow {

template <typename LeafLabel>
LeafOnlyBinarySPDecompositionTree<LeafLabel> wrap_series_split(LeafOnlyBinarySeriesSplit<LeafLabel> const &split) {
  return LeafOnlyBinarySPDecompositionTree<LeafLabel>{
    wrap_series_split(
      GenericBinarySeriesSplit<
        std::monostate,
        std::monostate,
        LeafLabel>{
        std::monostate{},
        split.pre.raw_tree,
        split.post.raw_tree,
      }
    ),
  };
}

template <typename LeafLabel>
LeafOnlyBinarySPDecompositionTree<LeafLabel> wrap_parallel_split(LeafOnlyBinaryParallelSplit<LeafLabel> const &split) {
  return LeafOnlyBinarySPDecompositionTree<LeafLabel>{
    wrap_parallel_split(
      GenericBinaryParallelSplit<
        std::monostate,
        std::monostate,
        LeafLabel>{
        std::monostate{},
        split.lhs.raw_tree,
        split.rhs.raw_tree,
      }
    ),
  };
}

} // namespace FlexFlow

#endif
