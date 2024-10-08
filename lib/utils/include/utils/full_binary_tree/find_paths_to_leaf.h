#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_FULL_BINARY_TREE_FIND_PATHS_TO_LEAF_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_FULL_BINARY_TREE_FIND_PATHS_TO_LEAF_H

#include "utils/containers/set_union.h"
#include "utils/containers/transform.h"
#include "utils/full_binary_tree/binary_tree_path.dtg.h"
#include "utils/full_binary_tree/binary_tree_path.h"
#include "utils/full_binary_tree/visit.h"
#include <unordered_set>

namespace FlexFlow {

template <typename Tree, typename Parent, typename Leaf>
std::unordered_set<BinaryTreePath> find_paths_to_leaf(
    Tree const &tree,
    FullBinaryTreeImplementation<Tree, Parent, Leaf> const &impl,
    Leaf const &needle) {
  auto visitor = FullBinaryTreeVisitor<std::unordered_set<BinaryTreePath>,
                                       Tree,
                                       Parent,
                                       Leaf>{
      [&](Parent const &parent) -> std::unordered_set<BinaryTreePath> {
        return set_union(
            transform(
                find_paths_to_leaf(impl.get_left_child(parent), impl, needle),
                [](BinaryTreePath const &path) {
                  return nest_inside_left_child(path);
                }),
            transform(
                find_paths_to_leaf(impl.get_right_child(parent), impl, needle),
                [](BinaryTreePath const &path) {
                  return nest_inside_right_child(path);
                }));
      },
      [&](Leaf const &leaf) -> std::unordered_set<BinaryTreePath> {
        if (leaf == needle) {
          return {binary_tree_root_path()};
        } else {
          return {};
        }
      },
  };

  return visit(tree, impl, visitor);
}

} // namespace FlexFlow

#endif
