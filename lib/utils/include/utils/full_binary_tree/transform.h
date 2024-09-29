#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_FULL_BINARY_TREE_TRANSFORM_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_FULL_BINARY_TREE_TRANSFORM_H

#include "utils/full_binary_tree/full_binary_tree.h"
#include "utils/full_binary_tree/get_left_child.h"
#include "utils/full_binary_tree/get_right_child.h"
#include "utils/overload.h"
#include "utils/full_binary_tree/visit.h"

namespace FlexFlow {

template <typename ParentLabel, 
          typename LeafLabel, 
          typename F, 
          typename ParentLabel2 = std::invoke_result_t<F, ParentLabel>,
          typename LeafLabel2 = std::invoke_result_t<F, LeafLabel>>
FullBinaryTreeParentNode<ParentLabel2, LeafLabel2> transform(FullBinaryTreeParentNode<ParentLabel, LeafLabel> const &t, F f) {
  return FullBinaryTreeParentNode<ParentLabel2, LeafLabel2>{
    transform(get_left_child(t), f),
    transform(get_right_child(t), f),
  };
}

template <typename ParentLabel, 
          typename LeafLabel, 
          typename F, 
          typename ParentLabel2 = std::invoke_result_t<F, ParentLabel>,
          typename LeafLabel2 = std::invoke_result_t<F, LeafLabel>>
FullBinaryTree<ParentLabel2, LeafLabel2> transform(FullBinaryTree<ParentLabel, LeafLabel> const &t, F f) {
  return visit<FullBinaryTree<ParentLabel2, LeafLabel2>>
    ( t,
      overload {
        [&](FullBinaryTreeParentNode<ParentLabel, LeafLabel> const &parent) {
          return FullBinaryTree<ParentLabel2, LeafLabel2>{
            transform(parent, f),
          };
        },
        [&](LeafLabel const &leaf) {
          return FullBinaryTree<ParentLabel2, LeafLabel2>{
            f(leaf),
          };
        }
      });
}

} // namespace FlexFlow

#endif
