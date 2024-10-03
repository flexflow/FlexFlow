#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_FULL_BINARY_TREE_REQUIRE_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_FULL_BINARY_TREE_REQUIRE_H

#include "utils/full_binary_tree/full_binary_tree.dtg.h"
#include "utils/full_binary_tree/full_binary_tree_parent_node.dtg.h"

namespace FlexFlow {

template <typename ParentLabel, typename LeafLabel>
FullBinaryTreeParentNode<ParentLabel, LeafLabel> require_parent_node(FullBinaryTree<ParentLabel, LeafLabel> const &t) {
  if (t.raw_tree.is_leaf()) {
    throw mk_runtime_error(fmt::format("require_parent_node called on leaf node {}", t));
  }

  return FullBinaryTreeParentNode<ParentLabel, LeafLabel>{
    t.raw_tree,
  };
}

template <typename ParentLabel, typename LeafLabel>
LeafLabel require_leaf(FullBinaryTree<ParentLabel, LeafLabel> const &t) {
  if (!t.raw_tree.is_leaf()) {
    throw mk_runtime_error(fmt::format("require_leaf called on non-leaf node {}", t));
  }

  return t.raw_tree.label.template get<LeafLabel>();
}

} // namespace FlexFlow

#endif
