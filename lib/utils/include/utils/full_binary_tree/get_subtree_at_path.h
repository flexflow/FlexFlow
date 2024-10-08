#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_FULL_BINARY_TREE_GET_SUBTREE_AT_PATH_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_FULL_BINARY_TREE_GET_SUBTREE_AT_PATH_H

#include "utils/full_binary_tree/binary_tree_path.dtg.h"
#include "utils/full_binary_tree/binary_tree_path.h"
#include "utils/full_binary_tree/get_child.h"
#include "utils/full_binary_tree/visit.h"
#include <optional>

namespace FlexFlow {

template <typename Tree, typename Parent, typename Leaf>
std::optional<Tree> get_subtree_at_path(
    Tree const &tree,
    FullBinaryTreeImplementation<Tree, Parent, Leaf> const &impl,
    BinaryTreePath const &p) {
  if (p == binary_tree_root_path()) {
    return tree;
  }

  auto visitor = FullBinaryTreeVisitor<std::optional<Tree>, Tree, Parent, Leaf>{
      [&](Parent const &parent) -> std::optional<Tree> {
        BinaryTreePathEntry curr = binary_tree_path_get_top_level(p);
        BinaryTreePath rest = binary_tree_path_get_non_top_level(p);

        return get_subtree_at_path(get_child(parent, impl, curr), impl, rest);
      },
      [](Leaf const &leaf) -> std::optional<Tree> { return std::nullopt; },
  };

  return visit(tree, impl, visitor);
}

} // namespace FlexFlow

#endif
