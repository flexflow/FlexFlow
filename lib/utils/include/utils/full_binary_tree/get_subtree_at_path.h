#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_FULL_BINARY_TREE_GET_SUBTREE_AT_PATH_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_FULL_BINARY_TREE_GET_SUBTREE_AT_PATH_H

#include "utils/full_binary_tree/binary_tree_path.dtg.h"
#include "utils/full_binary_tree/full_binary_tree.dtg.h"
#include "utils/containers/transform.h"
#include "utils/full_binary_tree/raw_full_binary_tree/algorithms.h"

namespace FlexFlow {

template <typename ParentLabel, typename LeafLabel>
std::optional<FullBinaryTree<ParentLabel, LeafLabel>> get_subtree_at_path(FullBinaryTree<ParentLabel, LeafLabel> const &t,
                                                                          BinaryTreePath const &p) {
  return transform(get_subtree_at_path(t.raw_tree, p),
                   [](RawBinaryTree const &raw) {
                     return FullBinaryTree<ParentLabel, LeafLabel>{raw};
                   });
}

} // namespace FlexFlow

#endif
