#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_FULL_BINARY_TREE_GET_NUM_TREE_NODES_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_FULL_BINARY_TREE_GET_NUM_TREE_NODES_H

#include "utils/full_binary_tree/full_binary_tree_implementation.dtg.h"
#include "utils/full_binary_tree/visit.h"

namespace FlexFlow {

template <typename Tree, typename Parent, typename Leaf>
int get_num_tree_nodes(
    Tree const &tree,
    FullBinaryTreeImplementation<Tree, Parent, Leaf> const &impl) {

  auto visitor = FullBinaryTreeVisitor<int, Tree, Parent, Leaf>{
      [&](Parent const &parent) -> int {
        return 1 + get_num_tree_nodes(impl.get_left_child(parent), impl) +
               get_num_tree_nodes(impl.get_right_child(parent), impl);
      },
      [](Leaf const &) -> int { return 1; },
  };

  return visit(tree, impl, visitor);
}

} // namespace FlexFlow

#endif
