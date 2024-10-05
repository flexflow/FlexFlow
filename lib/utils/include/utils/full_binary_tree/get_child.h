#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_FULL_BINARY_TREE_GET_CHILD_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_FULL_BINARY_TREE_GET_CHILD_H

#include "utils/exception.h"
#include "utils/full_binary_tree/binary_tree_path_entry.dtg.h"
#include "utils/full_binary_tree/full_binary_tree.h"
#include "utils/full_binary_tree/get_left_child.h"
#include "utils/full_binary_tree/get_right_child.h"
#include <fmt/format.h>

namespace FlexFlow {

template <typename ParentLabel, typename LeafLabel>
FullBinaryTree<ParentLabel, LeafLabel>
    get_child(FullBinaryTreeParentNode<ParentLabel, LeafLabel> const &t,
              BinaryTreePathEntry const &e) {
  switch (e) {
    case BinaryTreePathEntry::LEFT_CHILD:
      return get_left_child(t);
    case BinaryTreePathEntry::RIGHT_CHILD:
      return get_right_child(t);
    default:
      throw mk_runtime_error(
          fmt::format("Unhandled BinaryTreePathEntry value: {}", e));
  }
}

} // namespace FlexFlow

#endif
