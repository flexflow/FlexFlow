#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_FULL_BINARY_TREE_GET_NODE_TYPE_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_FULL_BINARY_TREE_GET_NODE_TYPE_H

#include "utils/full_binary_tree/full_binary_tree.dtg.h"
#include "utils/full_binary_tree/full_binary_tree_node_type.dtg.h"
#include "utils/overload.h"

namespace FlexFlow {

template <typename ParentLabel, typename LeafLabel>
FullBinaryTreeNodeType
    get_node_type(FullBinaryTree<ParentLabel, LeafLabel> const &tree) {
  return tree.template visit<FullBinaryTreeNodeType>(overload {
    [](FullBinaryTreeParentNode<ParentLabel, LeafLabel> const &) { return FullBinaryTreeNodeType::PARENT; },
    [](LeafLabel const &) { return FullBinaryTreeNodeType::LEAF; },
  });
}

} // namespace FlexFlow

#endif
