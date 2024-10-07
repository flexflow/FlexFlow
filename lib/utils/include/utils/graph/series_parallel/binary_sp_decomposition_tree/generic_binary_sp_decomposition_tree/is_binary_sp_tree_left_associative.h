#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIES_PARALLEL_BINARY_SP_DECOMPOSITION_TREE_GENERIC_BINARY_SP_DECOMPOSITION_TREE_IS_BINARY_SP_TREE_LEFT_ASSOCIATIVE_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIES_PARALLEL_BINARY_SP_DECOMPOSITION_TREE_GENERIC_BINARY_SP_DECOMPOSITION_TREE_IS_BINARY_SP_TREE_LEFT_ASSOCIATIVE_H

#include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree_implementation.dtg.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/visit.h"

namespace FlexFlow {

template <typename Tree, typename Series, typename Parallel, typename Leaf>
bool is_binary_sp_tree_left_associative(
    Tree const &tree,
    GenericBinarySPDecompositionTreeImplementation<Tree, Series, Parallel, Leaf> const &impl) { 

  auto visitor = GenericBinarySPDecompositionTreeVisitor<bool, Tree, Series, Parallel, Leaf>{
    [&](Series const &split) {
      return impl.get_node_type(impl.series_get_right_child(split)) != SPDecompositionTreeNodeType::SERIES &&
             is_binary_sp_tree_left_associative(impl.series_get_left_child(split), impl) &&
             is_binary_sp_tree_left_associative(impl.series_get_right_child(split), impl);
    },
    [&](Parallel const &split) {
      return impl.get_node_type(impl.parallel_get_right_child(split)) != SPDecompositionTreeNodeType::PARALLEL &&
             is_binary_sp_tree_left_associative(impl.parallel_get_left_child(split), impl) &&
             is_binary_sp_tree_left_associative(impl.parallel_get_right_child(split), impl);
    },
    [&](Leaf const &leaf) {
      return true;
    },
  };

  return visit(tree, impl, visitor);
}

} // namespace FlexFlow

#endif
