#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_FULL_BINARY_TREE_GET_LEAVES_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_FULL_BINARY_TREE_GET_LEAVES_H

#include "utils/containers/multiset_union.h"
#include "utils/full_binary_tree/full_binary_tree.h"
#include "utils/full_binary_tree/visit.h"
#include "utils/overload.h"
#include <unordered_set>

namespace FlexFlow {

template <typename ParentLabel, typename ChildLabel>
std::unordered_multiset<ChildLabel>
    get_leaves(FullBinaryTree<ParentLabel, ChildLabel> const &t) {
  return visit<std::unordered_set<ChildLabel>>(
      t,
      overload{
          [](FullBinaryTreeParentNode<ParentLabel, ChildLabel> const &parent) {
            return multiset_union(get_leaves(get_left_child(parent)),
                                  get_leaves(get_right_child(parent)));
          },
          [](ChildLabel const &leaf) {
            return std::unordered_multiset<ChildLabel>{leaf};
          }});
}

} // namespace FlexFlow

#endif
