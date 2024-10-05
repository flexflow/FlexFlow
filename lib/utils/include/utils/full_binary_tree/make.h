#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_FULL_BINARY_TREE_MAKE_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_FULL_BINARY_TREE_MAKE_H

#include "utils/full_binary_tree/full_binary_tree.h"

namespace FlexFlow {

template <typename ParentLabel, typename LeafLabel>
FullBinaryTree<ParentLabel, LeafLabel> make_full_binary_tree_parent(
    ParentLabel const &label,
    FullBinaryTree<ParentLabel, LeafLabel> const &lhs,
    FullBinaryTree<ParentLabel, LeafLabel> const &rhs) {
  return FullBinaryTree<ParentLabel, LeafLabel>{
      FullBinaryTreeParentNode<ParentLabel, LeafLabel>{
          label,
          lhs,
          rhs,
      },
  };
}

template <typename ParentLabel, typename LeafLabel>
FullBinaryTree<ParentLabel, LeafLabel>
    make_full_binary_tree_leaf(LeafLabel const &label) {
  return FullBinaryTree<ParentLabel, LeafLabel>{
      label,
  };
}

} // namespace FlexFlow

#endif
