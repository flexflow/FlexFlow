#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_FULL_BINARY_TREE_GET_LEAVES_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_FULL_BINARY_TREE_GET_LEAVES_H

#include "utils/containers/multiset_union.h"
#include "utils/full_binary_tree/full_binary_tree_visitor.dtg.h"
#include "utils/full_binary_tree/visit.h"
#include <unordered_set>

namespace FlexFlow {

template <typename Tree, typename Parent, typename Leaf>
std::unordered_multiset<Leaf>
    get_leaves(Tree const &tree,
               FullBinaryTreeImplementation<Tree, Parent, Leaf> const &impl) {

  auto visitor =
      FullBinaryTreeVisitor<std::unordered_multiset<Leaf>, Tree, Parent, Leaf>{
          [&](Parent const &parent) -> std::unordered_multiset<Leaf> {
            return multiset_union(
                get_leaves(impl.get_left_child(parent), impl),
                get_leaves(impl.get_right_child(parent), impl));
          },
          [](Leaf const &leaf) -> std::unordered_multiset<Leaf> {
            return {leaf};
          },
      };

  return visit(tree, impl, visitor);
}

} // namespace FlexFlow

#endif
