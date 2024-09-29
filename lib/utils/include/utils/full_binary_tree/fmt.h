#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_FULL_BINARY_TREE_FMT_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_FULL_BINARY_TREE_FMT_H

#include "utils/full_binary_tree/full_binary_tree.h"
#include "utils/full_binary_tree/get_left_child.h"
#include "utils/full_binary_tree/get_right_child.h"
#include "utils/full_binary_tree/visit.h"
#include "utils/overload.h"
#include <fmt/format.h>

namespace FlexFlow {

template <typename ParentLabel, typename LeafLabel>
std::string format_as(FullBinaryTreeParentNode<ParentLabel, LeafLabel> const &t) {
  return fmt::format("<{} ({} {})>",
                     t.label,
                     get_left_child(t),
                     get_right_child(t));
}

template <typename ParentLabel, typename LeafLabel>
std::string format_as(FullBinaryTree<ParentLabel, LeafLabel> const &t) {
  return visit<std::string>(
      t,
      overload{
          [](FullBinaryTreeParentNode<ParentLabel, LeafLabel> const &parent) {
            return fmt::to_string(parent);
          },
          [](LeafLabel const &leaf) {
            return fmt::format("{}", leaf);
          },
      });
}

} // namespace FlexFlow

#endif
