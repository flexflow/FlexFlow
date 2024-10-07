#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIES_PARALLEL_BINARY_SP_DECOMPOSITION_TREE_LEAF_ONLY_BINARY_SP_DECOMPOSITION_TREE_TRANSFORM_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIES_PARALLEL_BINARY_SP_DECOMPOSITION_TREE_LEAF_ONLY_BINARY_SP_DECOMPOSITION_TREE_TRANSFORM_H

// #include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/transform.h"
// #include "utils/graph/series_parallel/binary_sp_decomposition_tree/leaf_only_binary_sp_decomposition_tree/wrap.h"
// #include "utils/graph/series_parallel/binary_sp_decomposition_tree/leaf_only_binary_sp_decomposition_tree/leaf_only_binary_parallel_split.dtg.h"
// #include "utils/graph/series_parallel/binary_sp_decomposition_tree/leaf_only_binary_sp_decomposition_tree/leaf_only_binary_series_split.dtg.h"
// #include "utils/graph/series_parallel/binary_sp_decomposition_tree/leaf_only_binary_sp_decomposition_tree/leaf_only_binary_sp_decomposition_tree_visitor.dtg.h"

namespace FlexFlow {

// template <typename LeafLabel, typename LeafLabel2>
// LeafOnlyBinarySeriesSplit<LeafLabel2> transform(
//     LeafOnlyBinarySeriesSplit<LeafLabel> const &t,
//     LeafOnlyBinarySPDecompositionTreeVisitor<LeafLabel, LeafLabel2> const
//         &visitor) {
//   return transform(leaf_only_binary_sp_tree_wrap_series_split(t), visitor);
// }
//
// template <typename LeafLabel, typename LeafLabel2>
// LeafOnlyBinaryParallelSplit<LeafLabel2> transform(
//     LeafOnlyBinaryParallelSplit<LeafLabel> const &t,
//     LeafOnlyBinarySPDecompositionTreeVisitor<LeafLabel, LeafLabel2> const
//         &visitor) {
//   return transform(leaf_only_binary_sp_tree_wrap_parallel_split(t), visitor);
// }
//
// template <typename LeafLabel, typename LeafLabel2>
// LeafOnlyBinarySPDecompositionTree<LeafLabel2> transform(
//     LeafOnlyBinarySPDecompositionTree<LeafLabel> const &t,
//     LeafOnlyBinarySPDecompositionTreeVisitor<LeafLabel, LeafLabel2> const
//         &visitor) {
//   using GenericVisitor = GenericBinarySPDecompositionTreeTransformVisitor<std::monostate,
//                                                                  std::monostate,
//                                                                  LeafLabel,
//                                                                  std::monostate,
//                                                                  std::monostate,
//                                                                  LeafLabel2>;
//
//   GenericVisitor generic_visitor = GenericVisitor{
//       [&](std::monostate const &x) { return x; },
//       [&](std::monostate const &x) { return x; },
//       [&](LeafLabel const &t) { return visitor.leaf_func(t); },
//   };
//
//   return LeafOnlyBinarySPDecompositionTree<LeafLabel2>{
//       transform(t.raw_tree, generic_visitor),
//   };
// }

} // namespace FlexFlow

#endif
