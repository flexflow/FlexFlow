#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_FULL_BINARY_TREE_FIND_PATHS_TO_LEAF_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_FULL_BINARY_TREE_FIND_PATHS_TO_LEAF_H

#include "utils/containers/set_union.h"
#include "utils/containers/transform.h"
#include "utils/full_binary_tree/binary_tree_path.dtg.h"
#include "utils/full_binary_tree/binary_tree_path.h"
#include "utils/full_binary_tree/full_binary_tree.h"
#include "utils/full_binary_tree/visit.h"
#include "utils/overload.h"
#include <unordered_set>

namespace FlexFlow {

template <typename ParentLabel, typename LeafLabel>
std::unordered_set<BinaryTreePath>
    find_paths_to_leaf(FullBinaryTree<ParentLabel, LeafLabel> const &tree,
                       LeafLabel const &leaf) {
  return visit<std::unordered_set<BinaryTreePath>>(
      tree,
      overload{
          [&](LeafLabel const &l) -> std::unordered_set<BinaryTreePath> {
            if (l == leaf) {
              return {binary_tree_root_path()};
            } else {
              return {};
            }
          },
          [&](FullBinaryTreeParentNode<ParentLabel, LeafLabel> const &parent) {
            return set_union(
                transform(find_paths_to_leaf(get_left_child(parent), leaf),
                          nest_inside_left_child),
                transform(find_paths_to_leaf(get_right_child(parent), leaf),
                          nest_inside_right_child));
          }});
}

} // namespace FlexFlow

#endif
