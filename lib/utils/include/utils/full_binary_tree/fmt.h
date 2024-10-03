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
  auto visitor = FullBinaryTreeVisitor<std::string, ParentLabel, LeafLabel>{
    [](FullBinaryTreeParentNode<ParentLabel, LeafLabel> const &parent) {
      return fmt::to_string(parent);
    },
    [](LeafLabel const &leaf) {
      return fmt::format("{}", leaf);
    },
  };

  return visit(t, visitor);
}

template <typename ParentLabel, typename LeafLabel>
std::ostream &operator<<(std::ostream &s, FullBinaryTreeParentNode<ParentLabel, LeafLabel> const &t) {
  return (s << fmt::to_string(t));
}

template <typename ParentLabel, typename LeafLabel>
std::ostream &operator<<(std::ostream &s, FullBinaryTree<ParentLabel, LeafLabel> const &t) {
  return (s << fmt::to_string(t));
}

} // namespace FlexFlow

#endif
