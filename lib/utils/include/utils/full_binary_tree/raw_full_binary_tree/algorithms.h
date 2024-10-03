#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_FULL_BINARY_TREE_RAW_FULL_BINARY_TREE_ALGORITHMS_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_FULL_BINARY_TREE_RAW_FULL_BINARY_TREE_ALGORITHMS_H

#include "utils/full_binary_tree/binary_tree_path.dtg.h"
#include "utils/full_binary_tree/binary_tree_path_entry.dtg.h"
#include "utils/full_binary_tree/full_binary_tree_node_type.dtg.h"
#include "utils/full_binary_tree/raw_full_binary_tree/raw_binary_tree.h"
#include <unordered_set>

namespace FlexFlow {

RawBinaryTree get_child(RawBinaryTree const &, BinaryTreePathEntry const &);
std::unordered_set<BinaryTreePath> get_all_leaf_paths(RawBinaryTree const &);
std::unordered_set<BinaryTreePath> find_paths_to_leaf(RawBinaryTree const &, any_value_type const &leaf);
std::unordered_multiset<any_value_type> get_leaves(RawBinaryTree const &);
FullBinaryTreeNodeType get_node_type(RawBinaryTree const &);
std::optional<RawBinaryTree> get_subtree_at_path(RawBinaryTree const &, BinaryTreePath const &);

} // namespace FlexFlow

#endif
