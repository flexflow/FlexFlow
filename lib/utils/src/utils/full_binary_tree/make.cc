#include "utils/full_binary_tree/make.h"

namespace FlexFlow {

template FullBinaryTree<int, int>
    make_full_binary_tree_parent(int const &,
                                 FullBinaryTree<int, int> const &,
                                 FullBinaryTree<int, int> const &);
template FullBinaryTree<int, int> make_full_binary_tree_leaf(int const &);

} // namespace FlexFlow
