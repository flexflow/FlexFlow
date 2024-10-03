#include "utils/full_binary_tree/raw_full_binary_tree/algorithms.h"
#include "utils/full_binary_tree/binary_tree_path.h"
#include "utils/containers/transform.h"
#include "utils/containers/set_union.h"
#include "utils/containers/multiset_union.h"

namespace FlexFlow {

RawBinaryTree get_child(RawBinaryTree const &t, BinaryTreePathEntry const &e) {
  if (e == BinaryTreePathEntry::LEFT_CHILD) {
    return t.left_child();
  } else {
    assert (e == BinaryTreePathEntry::RIGHT_CHILD);
    return t.right_child();
  }
}

std::unordered_set<BinaryTreePath> get_all_leaf_paths(RawBinaryTree const &t) {
  if (t.is_leaf()) {
    return {binary_tree_root_path()};
  } else {
    return set_union(
      transform(get_all_leaf_paths(t.left_child()),
                [](BinaryTreePath const &path) {
                  return nest_inside_left_child(path);
                }),
      transform(get_all_leaf_paths(t.right_child()),
                [](BinaryTreePath const &path) {
                  return nest_inside_right_child(path);
                }));
  }
}

std::unordered_set<BinaryTreePath> find_paths_to_leaf(RawBinaryTree const &t, any_value_type const &leaf) {
  if (t.is_leaf()) {
    if (t.label == leaf) {
      return {binary_tree_root_path()};
    } else {
      return {};
    }
  } else {
    return set_union(
      transform(find_paths_to_leaf(t.left_child(), leaf),
                [](BinaryTreePath const &path) {
                  return nest_inside_left_child(path);
                }),
      transform(find_paths_to_leaf(t.right_child(), leaf),
                [](BinaryTreePath const &path) {
                  return nest_inside_right_child(path);
                }));
  }
}

std::unordered_multiset<any_value_type> get_leaves(RawBinaryTree const &t) {
  if (t.is_leaf()) {
    return {t.label};
  } else {
    return multiset_union(get_leaves(t.left_child()), get_leaves(t.right_child()));
  }
}

FullBinaryTreeNodeType get_node_type(RawBinaryTree const &t) {
  if (t.is_leaf()) {
    return FullBinaryTreeNodeType::LEAF;
  } else {
    return FullBinaryTreeNodeType::PARENT;
  }
}

std::optional<RawBinaryTree> get_subtree_at_path(RawBinaryTree const &t, BinaryTreePath const &p) {
  if (p == binary_tree_root_path()) {
    return t;
  } else if (t.is_leaf()) {
    return std::nullopt;
  } else {
    BinaryTreePathEntry curr = binary_tree_path_get_top_level(p);
    BinaryTreePath rest = binary_tree_path_get_non_top_level(p);

    return get_subtree_at_path(get_child(t, curr), rest);
  }
}

} // namespace FlexFlow
