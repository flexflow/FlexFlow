#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_FULL_BINARY_TREE_VISIT_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_FULL_BINARY_TREE_VISIT_H

#include "utils/full_binary_tree/full_binary_tree.h"
#include "utils/exception.h"

namespace FlexFlow {

template <typename Result, typename F, typename ParentLabel, typename LeafLabel>
Result visit(FullBinaryTree<ParentLabel, LeafLabel> const &tt, F f) {
  if (std::holds_alternative<FullBinaryTreeParentNode<ParentLabel, LeafLabel>>(tt.root)) {
    return f(std::get<FullBinaryTreeParentNode<ParentLabel, LeafLabel>>(tt.root));
  } else if (std::holds_alternative<LeafLabel>(tt.root)) {
    return f(std::get<LeafLabel>(tt.root));
  } else {
    throw mk_runtime_error(
        "Unexpected case in visit(FullBinaryTree)");
  }
}


} // namespace FlexFlow

#endif
