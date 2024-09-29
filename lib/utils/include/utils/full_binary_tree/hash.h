#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_FULL_BINARY_TREE_HASH_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_FULL_BINARY_TREE_HASH_H

#include "utils/full_binary_tree/full_binary_tree.h"
#include "utils/hash-utils.h"
#include "utils/hash/tuple.h"

namespace std {

template <typename ParentLabel, typename LeafLabel>
struct hash<::FlexFlow::FullBinaryTreeParentNode<ParentLabel, LeafLabel>> {
  size_t operator()(::FlexFlow::FullBinaryTreeParentNode<ParentLabel, LeafLabel> const &t) const {
    return get_std_hash(t.tie());
  }
};

template <typename ParentLabel, typename LeafLabel>
struct hash<::FlexFlow::FullBinaryTree<ParentLabel, LeafLabel>> {
  size_t operator()(::FlexFlow::FullBinaryTree<ParentLabel, LeafLabel> const &t) const {
    return get_std_hash(t.tie());
  }
};

} // namespace FlexFlow

#endif
