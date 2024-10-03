#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_FULL_BINARY_TREE_GET_CHILD_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_FULL_BINARY_TREE_GET_CHILD_H

#include "utils/full_binary_tree/full_binary_tree.dtg.h"
#include "utils/full_binary_tree/full_binary_tree_parent_node.dtg.h"
#include "utils/full_binary_tree/raw_full_binary_tree/algorithms.h"

namespace FlexFlow {

template <typename ParentLabel, typename LeafLabel>
FullBinaryTree<ParentLabel, LeafLabel> get_child(FullBinaryTreeParentNode<ParentLabel, LeafLabel> const &t,
                                                 BinaryTreePathEntry const &e) {
  return FullBinaryTreeParentNode<ParentLabel, LeafLabel>{
    get_child(t.raw_tree, e),
  };
}

} // namespace FlexFlow

#endif
