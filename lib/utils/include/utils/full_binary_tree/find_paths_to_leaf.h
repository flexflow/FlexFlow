#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_FULL_BINARY_TREE_FIND_PATHS_TO_LEAF_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_FULL_BINARY_TREE_FIND_PATHS_TO_LEAF_H

#include "utils/full_binary_tree/binary_tree_path.dtg.h"
#include "utils/full_binary_tree/full_binary_tree.dtg.h"
#include <unordered_set>

namespace FlexFlow {

template <typename ParentLabel, typename LeafLabel>
std::unordered_set<BinaryTreePath> find_paths_to_leaf(FullBinaryTree<ParentLabel, LeafLabel> const &tree,
                                                      LeafLabel const &leaf) {
  return find_paths_to_leaf(tree.raw_tree, make_any_value_type<LeafLabel>(leaf));
}

} // namespace FlexFlow

#endif
