#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIES_PARALLEL_BINARY_SP_DECOMPOSITION_TREE_LEAF_ONLY_BINARY_SP_DECOMPOSITION_TREE_REQUIRE_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIES_PARALLEL_BINARY_SP_DECOMPOSITION_TREE_LEAF_ONLY_BINARY_SP_DECOMPOSITION_TREE_REQUIRE_H

#include "utils/graph/series_parallel/binary_sp_decomposition_tree/leaf_only_binary_sp_decomposition_tree/leaf_only_binary_sp_decomposition_tree.dtg.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/leaf_only_binary_sp_decomposition_tree/leaf_only_binary_series_split.dtg.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/leaf_only_binary_sp_decomposition_tree/leaf_only_binary_parallel_split.dtg.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/generic_binary_series_split.dtg.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/generic_binary_parallel_split.dtg.h"

namespace FlexFlow {

template <typename LeafLabel>
LeafOnlyBinarySeriesSplit<LeafLabel>
  require_series(LeafOnlyBinarySPDecompositionTree<LeafLabel> const &t) {
  GenericBinarySeriesSplit<
    LeafOnlyBinarySeriesSplitLabel, 
    LeafOnlyBinaryParallelSplitLabel,
    LeafLabel> raw = 
      require_series(t.raw_tree);

  return LeafOnlyBinarySeriesSplit<LeafLabel>{
    LeafOnlyBinarySeriesSplitLabel{},
    LeafOnlyBinarySPDecompositionTree<LeafLabel>{raw.pre},
    LeafOnlyBinarySPDecompositionTree<LeafLabel>{raw.post},
  };
}

template <typename LeafLabel>
LeafOnlyBinaryParallelSplit<LeafLabel>
  require_parallel(LeafOnlyBinarySPDecompositionTree<LeafLabel> const &t) {
  GenericBinarySeriesSplit<
    LeafOnlyBinarySeriesSplitLabel, 
    LeafOnlyBinaryParallelSplitLabel,
    LeafLabel> raw = 
      require_series(t.raw_tree);

  return LeafOnlyBinarySeriesSplit<LeafLabel>{
    LeafOnlyBinaryParallelSplitLabel{},
    LeafOnlyBinarySPDecompositionTree<LeafLabel>{raw.pre},
    LeafOnlyBinarySPDecompositionTree<LeafLabel>{raw.post},
  };
}

template <typename LeafLabel>
LeafLabel require_leaf(LeafOnlyBinarySPDecompositionTree<LeafLabel> const &t) {
  return require_leaf(t.raw_tree);
}


} // namespace FlexFlow

#endif
