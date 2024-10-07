#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIES_PARALLEL_BINARY_SP_DECOMPOSITION_TREE_LEAF_ONLY_BINARY_SP_DECOMPOSITION_TREE_REQUIRE_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIES_PARALLEL_BINARY_SP_DECOMPOSITION_TREE_LEAF_ONLY_BINARY_SP_DECOMPOSITION_TREE_REQUIRE_H

// #include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/generic_binary_parallel_split.dtg.h"
// #include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/generic_binary_series_split.dtg.h"
// #include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/require.h"
// #include "utils/graph/series_parallel/binary_sp_decomposition_tree/leaf_only_binary_sp_decomposition_tree/leaf_only_binary_parallel_split.dtg.h"
// #include "utils/graph/series_parallel/binary_sp_decomposition_tree/leaf_only_binary_sp_decomposition_tree/leaf_only_binary_series_split.dtg.h"
// #include "utils/graph/series_parallel/binary_sp_decomposition_tree/leaf_only_binary_sp_decomposition_tree/leaf_only_binary_sp_decomposition_tree.dtg.h"

namespace FlexFlow {

// template <typename LeafLabel>
// LeafOnlyBinarySeriesSplit<LeafLabel> require_leaf_only_binary_series_split(
//     LeafOnlyBinarySPDecompositionTree<LeafLabel> const &t) {
//   GenericBinarySeriesSplit<std::monostate, std::monostate, LeafLabel> raw =
//       require_generic_binary_series_split(t.raw_tree);
//
//   return LeafOnlyBinarySeriesSplit<LeafLabel>{
//       LeafOnlyBinarySPDecompositionTree<LeafLabel>{raw.lhs},
//       LeafOnlyBinarySPDecompositionTree<LeafLabel>{raw.rhs},
//   };
// }
//
// template <typename LeafLabel>
// LeafOnlyBinaryParallelSplit<LeafLabel> require_leaf_only_binary_parallel_split(
//     LeafOnlyBinarySPDecompositionTree<LeafLabel> const &t) {
//   GenericBinaryParallelSplit<std::monostate, std::monostate, LeafLabel> raw =
//       require_generic_binary_parallel_split(t.raw_tree);
//
//   return LeafOnlyBinaryParallelSplit<LeafLabel>{
//       LeafOnlyBinarySPDecompositionTree<LeafLabel>{raw.lhs},
//       LeafOnlyBinarySPDecompositionTree<LeafLabel>{raw.rhs},
//   };
// }
//
// template <typename LeafLabel>
// LeafLabel require_leaf_only_binary_leaf(
//     LeafOnlyBinarySPDecompositionTree<LeafLabel> const &t) {
//   return require_generic_binary_leaf(t.raw_tree);
// }
//
} // namespace FlexFlow

#endif
