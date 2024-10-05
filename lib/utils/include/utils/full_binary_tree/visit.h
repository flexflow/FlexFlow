#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_FULL_BINARY_TREE_VISIT_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_FULL_BINARY_TREE_VISIT_H

#include "utils/exception.h"
#include "utils/full_binary_tree/full_binary_tree_visitor.dtg.h"
#include "utils/full_binary_tree/get_node_type.h"
#include "utils/full_binary_tree/require.h"

namespace FlexFlow {

template <typename Result, typename F, typename ParentLabel, typename LeafLabel>
Result visit(FullBinaryTree<ParentLabel, LeafLabel> const &tt, F f) {
  auto visitor = FullBinaryTreeVisitor<Result, ParentLabel, LeafLabel>{f, f};

  return visit(tt, visitor);
}

template <typename Result, typename ParentLabel, typename LeafLabel>
Result visit(FullBinaryTree<ParentLabel, LeafLabel> const &t,
             FullBinaryTreeVisitor<Result, ParentLabel, LeafLabel> const &v) {
  FullBinaryTreeNodeType node_type = get_node_type(t);
  switch (node_type) {
    case FullBinaryTreeNodeType::PARENT:
      return v.parent_func(require_full_binary_tree_parent_node(t));
    case FullBinaryTreeNodeType::LEAF:
      return v.leaf_func(require_full_binary_tree_leaf(t));
    default:
      throw mk_runtime_error(
          fmt::format("Unhandled FullBinaryTreeNodeType value: {}", node_type));
  }
}

} // namespace FlexFlow

#endif
