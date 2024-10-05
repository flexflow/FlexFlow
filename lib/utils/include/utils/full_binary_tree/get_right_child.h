#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_FULL_BINARY_TREE_GET_RIGHT_CHILD_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_FULL_BINARY_TREE_GET_RIGHT_CHILD_H

#include "utils/full_binary_tree/full_binary_tree.h"

namespace FlexFlow {

template <typename ParentLabel, typename LeafLabel>
FullBinaryTree<ParentLabel, LeafLabel> const &
    get_right_child(FullBinaryTreeParentNode<ParentLabel, LeafLabel> const &t) {
  return *t.right_child_ptr;
}

} // namespace FlexFlow

#endif
