#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIES_PARALLEL_BINARY_SP_DECOMPOSITION_TREE_LEAF_ONLY_BINARY_SP_DECOMPOSITION_TREE_LEAF_ONLY_BINARY_SP_DECOMPOSITION_TREE_VISITOR_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIES_PARALLEL_BINARY_SP_DECOMPOSITION_TREE_LEAF_ONLY_BINARY_SP_DECOMPOSITION_TREE_LEAF_ONLY_BINARY_SP_DECOMPOSITION_TREE_VISITOR_H

// #include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree_visitor.dtg.h"
// #include "utils/graph/series_parallel/binary_sp_decomposition_tree/leaf_only_binary_sp_decomposition_tree/leaf_only_binary_sp_decomposition_tree_visitor.dtg.h"

namespace FlexFlow {

// template <typename ReturnType, typename LeafLabel>
// GenericBinarySPDecompositionTreeVisitor<ReturnType, std::monostate, std::monostate, LeafLabel> 
//   generic_visitor_from_leaf_only_visitor(LeafOnlyBinarySPDecompositionTreeVisitor<ReturnType, LeafLabel> const &leaf_only) {
//   return GenericBinarySPDecompositionTreeVisitor<ReturnType, std::monostate, std::monostate, LeafLabel>{
//     [leaf_only](GenericBinarySeriesSplit<std::monostate, std::monostate, LeafLabel> const &split) {
//       return leaf_only.series_func(
//         LeafOnlyBinarySeriesSplit<LeafLabel>{
//           LeafOnlyBinarySPDecompositionTree<LeafLabel>{split.lhs},
//           LeafOnlyBinarySPDecompositionTree<LeafLabel>{split.rhs},
//         });
//     },
//     [leaf_only](GenericBinaryParallelSplit<std::monostate, std::monostate, LeafLabel> const &split) {
//       return leaf_only.parallel_func(
//         LeafOnlyBinaryParallelSplit<LeafLabel>{
//           LeafOnlyBinarySPDecompositionTree<LeafLabel>{split.lhs},
//           LeafOnlyBinarySPDecompositionTree<LeafLabel>{split.rhs},
//         });
//     },
//     [leaf_only](LeafLabel const &leaf_label) {
//       return leaf_only.leaf_func(leaf_label);
//     },
//   };
// }

} // namespace FlexFlow

#endif
