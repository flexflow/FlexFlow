#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_FULL_BINARY_TREE_GET_SUBTREE_AT_PATH_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_FULL_BINARY_TREE_GET_SUBTREE_AT_PATH_H

#include "utils/full_binary_tree/binary_tree_path.dtg.h"
#include "utils/full_binary_tree/binary_tree_path.h"
#include "utils/full_binary_tree/full_binary_tree.h"
#include "utils/full_binary_tree/get_child.h"
#include "utils/full_binary_tree/visit.h"
#include "utils/overload.h"
#include <optional>

namespace FlexFlow {

template <typename ParentLabel, typename LeafLabel>
std::optional<FullBinaryTree<ParentLabel, LeafLabel>> get_subtree_at_path(FullBinaryTree<ParentLabel, LeafLabel> const &t,
                                                                          BinaryTreePath const &p) {
  if (p == binary_tree_root_path()) {
    return t;
  }

  return visit<std::optional<FullBinaryTree<ParentLabel, LeafLabel>>>(
    t,
    overload {
      [&](FullBinaryTreeParentNode<ParentLabel, LeafLabel> const &parent) {
        BinaryTreePathEntry curr = binary_tree_path_get_top_level(p);
        BinaryTreePath rest = binary_tree_path_get_non_top_level(p);

        return get_subtree_at_path(get_child(parent, curr), rest);
      },
      [&](LeafLabel const &leaf) {
        return std::nullopt;
      }
    });
}

} // namespace FlexFlow

#endif
