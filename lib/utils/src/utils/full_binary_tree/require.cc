#include "utils/full_binary_tree/require.h"

namespace FlexFlow {

template 
  FullBinaryTreeParentNode<int, int> const &
    require_full_binary_tree_parent_node(FullBinaryTree<int, int> const &);
template
  int const &require_full_binary_tree_leaf(FullBinaryTree<int, int> const &);

} // namespace FlexFlow
