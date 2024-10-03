#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_FULL_BINARY_TREE_GET_NODE_TYPE_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_FULL_BINARY_TREE_GET_NODE_TYPE_H

#include "utils/full_binary_tree/full_binary_tree.dtg.h"
#include "utils/full_binary_tree/raw_full_binary_tree/algorithms.h"

namespace FlexFlow {

template <typename ParentLabel, typename LeafLabel>
FullBinaryTreeNodeType get_node_type(FullBinaryTree<ParentLabel, LeafLabel> const &t) {
  return get_node_type(t.raw_tree);
}

} // namespace FlexFlow

#endif
