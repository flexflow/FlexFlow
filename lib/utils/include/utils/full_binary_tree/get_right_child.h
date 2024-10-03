#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_FULL_BINARY_TREE_GET_RIGHT_CHILD_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_FULL_BINARY_TREE_GET_RIGHT_CHILD_H

#include "utils/full_binary_tree/full_binary_tree.dtg.h"
#include "utils/full_binary_tree/full_binary_tree_parent_node.dtg.h"

namespace FlexFlow {

template <typename ParentLabel, typename LeafLabel>
FullBinaryTree<ParentLabel, LeafLabel> get_right_child(FullBinaryTreeParentNode<ParentLabel, LeafLabel> const &t) {
  return FullBinaryTree<ParentLabel, LeafLabel>{
    t.raw_tree.right_child(),
  };
}

} // namespace FlexFlow

#endif
