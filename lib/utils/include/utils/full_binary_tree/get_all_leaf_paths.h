#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_FULL_BINARY_TREE_GET_ALL_LEAF_PATHS_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_FULL_BINARY_TREE_GET_ALL_LEAF_PATHS_H

#include "utils/full_binary_tree/binary_tree_path.dtg.h"
#include "utils/full_binary_tree/full_binary_tree.dtg.h"
#include <unordered_set>
#include "utils/full_binary_tree/raw_full_binary_tree/algorithms.h"

namespace FlexFlow {

template <typename ParentLabel, typename LeafLabel>
std::unordered_set<BinaryTreePath> get_all_leaf_paths(FullBinaryTree<ParentLabel, LeafLabel> const &tree) {
  return get_all_leaf_paths(tree.raw_tree);
}

} // namespace FlexFlow

#endif
