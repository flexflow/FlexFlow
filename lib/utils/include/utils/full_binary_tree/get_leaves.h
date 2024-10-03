#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_FULL_BINARY_TREE_GET_LEAVES_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_FULL_BINARY_TREE_GET_LEAVES_H

#include "utils/full_binary_tree/full_binary_tree.dtg.h"
#include "utils/full_binary_tree/raw_full_binary_tree/algorithms.h"

namespace FlexFlow {

template <typename ParentLabel, typename LeafLabel>
std::unordered_multiset<LeafLabel>
  get_leaves(FullBinaryTree<ParentLabel, LeafLabel> const &t) {
  return transform(get_leaves(t.raw_tree), [](any_value_type const &v) { return v.get<LeafLabel>(); });
}

} // namespace FlexFlow

#endif
