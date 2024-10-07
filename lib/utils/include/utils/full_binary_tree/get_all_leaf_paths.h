#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_FULL_BINARY_TREE_GET_ALL_LEAF_PATHS_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_FULL_BINARY_TREE_GET_ALL_LEAF_PATHS_H

#include "utils/containers/set_union.h"
#include "utils/containers/transform.h"
#include "utils/full_binary_tree/binary_tree_path.dtg.h"
#include "utils/full_binary_tree/binary_tree_path.h"
#include "utils/full_binary_tree/visit.h"
#include "utils/overload.h"
#include <unordered_set>

namespace FlexFlow {

template <typename Tree, typename Parent, typename Leaf>
std::unordered_set<BinaryTreePath>
    get_all_leaf_paths(Tree const &tree,
                       FullBinaryTreeImplementation<Tree, Parent, Leaf> const &impl) {
  auto visitor = FullBinaryTreeVisitor<std::unordered_set<BinaryTreePath>, Tree, Parent, Leaf>{
    [&](Parent const &parent) -> std::unordered_set<BinaryTreePath> {
      return set_union(
          transform(get_all_leaf_paths(impl.get_left_child(parent), impl),
                    [](BinaryTreePath const &path) {
                      return nest_inside_left_child(path);
                    }),
          transform(get_all_leaf_paths(impl.get_right_child(parent), impl),
                    [](BinaryTreePath const &path) {
                      return nest_inside_right_child(path);
                    }));
    },
    [&](Leaf const &leaf) -> std::unordered_set<BinaryTreePath> {
      return {binary_tree_root_path()};
    },
  };

  return visit(tree, impl, visitor);
}

} // namespace FlexFlow

#endif
