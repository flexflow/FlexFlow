#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_FULL_BINARY_TREE_REQUIRE_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_FULL_BINARY_TREE_REQUIRE_H

#include "utils/full_binary_tree/full_binary_tree.dtg.h"
#include "utils/full_binary_tree/full_binary_tree_parent_node.dtg.h"

namespace FlexFlow {

template <typename ParentLabel, typename LeafLabel>
FullBinaryTreeParentNode<ParentLabel, LeafLabel> const &
    require_full_binary_tree_parent_node(
        FullBinaryTree<ParentLabel, LeafLabel> const &t) {
  return t.template get<FullBinaryTreeParentNode<ParentLabel, LeafLabel>>();
}

template <typename ParentLabel, typename LeafLabel>
LeafLabel const &require_full_binary_tree_leaf(
    FullBinaryTree<ParentLabel, LeafLabel> const &t) {
  return t.template get<LeafLabel>();
}

} // namespace FlexFlow

#endif
