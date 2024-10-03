#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_FULL_BINARY_TREE_GET_NODE_TYPE_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_FULL_BINARY_TREE_GET_NODE_TYPE_H

#include "utils/full_binary_tree/full_binary_tree.h"
#include "utils/full_binary_tree/full_binary_tree_node_type.dtg.h"

namespace FlexFlow {

template <typename ParentLabel, typename LeafLabel>
FullBinaryTreeNodeType get_node_type(FullBinaryTree<ParentLabel, LeafLabel> const &t) {
  if (std::holds_alternative<LeafLabel>(t.root)) {
    return FullBinaryTreeNodeType::LEAF;
  } else {
    bool is_parent = std::holds_alternative<FullBinaryTreeParentNode<ParentLabel, LeafLabel>>(t.root);
    assert (is_parent);

    return FullBinaryTreeNodeType::PARENT;
  }
}

} // namespace FlexFlow

#endif
