#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_FULL_BINARY_TREE_GET_LABEL_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_FULL_BINARY_TREE_GET_LABEL_H

#include "utils/full_binary_tree/full_binary_tree.h"

namespace FlexFlow {

template <typename ParentLabel, typename LeafLabel>
ParentLabel get_full_binary_tree_parent_label(
    FullBinaryTreeParentNode<ParentLabel, LeafLabel> const &p) {
  return p.label;
}

} // namespace FlexFlow

#endif
