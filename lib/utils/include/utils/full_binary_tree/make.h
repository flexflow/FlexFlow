#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_FULL_BINARY_TREE_MAKE_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_FULL_BINARY_TREE_MAKE_H

#include "utils/full_binary_tree/full_binary_tree.dtg.h"

namespace FlexFlow {

template <typename ParentLabel, typename LeafLabel>
FullBinaryTree<ParentLabel, LeafLabel> make_full_binary_tree_parent(ParentLabel const &label, 
                                                               FullBinaryTree<ParentLabel, LeafLabel> const &lhs,
                                                               FullBinaryTree<ParentLabel, LeafLabel> const &rhs) {
  return FullBinaryTree<ParentLabel, LeafLabel>{
    raw_binary_tree_make_parent(make_any_value_type<ParentLabel>(label), lhs.raw_tree, rhs.raw_tree),
  };
}

template <typename ParentLabel, typename LeafLabel>
FullBinaryTree<ParentLabel, LeafLabel> make_full_binary_tree_leaf(LeafLabel const &label) {
  return FullBinaryTree<ParentLabel, LeafLabel>{
    raw_binary_tree_make_leaf(make_any_value_type<LeafLabel>(label)),
  };
}

} // namespace FlexFlow

#endif
