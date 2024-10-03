#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_FULL_BINARY_TREE_GET_ALL_LEAF_PATHS_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_FULL_BINARY_TREE_GET_ALL_LEAF_PATHS_H

#include "utils/full_binary_tree/binary_tree_path.dtg.h"
#include "utils/full_binary_tree/binary_tree_path.h"
#include "utils/full_binary_tree/full_binary_tree.h"
#include "utils/full_binary_tree/visit.h"
#include <unordered_set>
#include "utils/overload.h"
#include "utils/containers/set_union.h"
#include "utils/containers/transform.h"

namespace FlexFlow {

template <typename ParentLabel, typename LeafLabel>
std::unordered_set<BinaryTreePath> get_all_leaf_paths(FullBinaryTree<ParentLabel, LeafLabel> const &tree) {
  return visit<std::unordered_set<BinaryTreePath>>
    (tree,
     overload {
      [](LeafLabel const &) {
        return std::unordered_set{binary_tree_root_path()};
      },
      [](FullBinaryTreeParentNode<ParentLabel, LeafLabel> const &parent) {
        return set_union(
          transform(get_all_leaf_paths(get_left_child(parent)), 
                    [](BinaryTreePath const &path) {
                      return nest_inside_left_child(path);
                    }),
          transform(get_all_leaf_paths(get_right_child(parent)), 
                    [](BinaryTreePath const &path) {
                      return nest_inside_right_child(path);
                    }));
      }
     });
}

} // namespace FlexFlow

#endif
