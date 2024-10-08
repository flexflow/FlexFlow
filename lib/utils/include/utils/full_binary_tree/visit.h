#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_FULL_BINARY_TREE_VISIT_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_FULL_BINARY_TREE_VISIT_H

#include "utils/exception.h"
#include "utils/full_binary_tree/full_binary_tree_visitor.dtg.h"
#include "utils/full_binary_tree/full_binary_tree_implementation.dtg.h"

namespace FlexFlow {

template <typename Result, typename Tree, typename Parent, typename Leaf>
Result visit(Tree const &tree, 
             FullBinaryTreeImplementation<Tree, Parent, Leaf> const &impl,
             FullBinaryTreeVisitor<Result, Tree, Parent, Leaf> const &visitor) {
  if (impl.is_leaf(tree)) {
    return visitor.leaf_func(impl.require_leaf(tree));
  } else {
    return visitor.parent_func(impl.require_parent(tree));
  }
}

} // namespace FlexFlow

#endif
