#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_FULL_BINARY_TREE_GET_NODE_TYPE_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_FULL_BINARY_TREE_GET_NODE_TYPE_H

#include "utils/overload.h"
#include "utils/full_binary_tree/full_binary_tree.h"
#include "utils/full_binary_tree/visit.h"
#include "utils/full_binary_tree/full_binary_tree_node_type.dtg.h"

namespace FlexFlow {

template <typename ParentLabel, typename LeafLabel>
FullBinaryTreeNodeType get_node_type(FullBinaryTree<ParentLabel, LeafLabel> const &t) {
  return visit<FullBinaryTreeNodeType>(
    t,
    overload {
      [](FullBinaryTreeParentNode<ParentLabel, LeafLabel> const &) {
        return FullBinaryTreeNodeType::PARENT; 
      },
      [](LeafLabel const &) {
        return FullBinaryTreeNodeType::LEAF;
      }
    });
}

} // namespace FlexFlow

#endif
