#include "utils/full_binary_tree/fmt.h"

namespace FlexFlow {

template std::string format_as(FullBinaryTreeParentNode<int, int> const &);
template std::string format_as(FullBinaryTree<int, int> const &);
template std::ostream &operator<<(std::ostream &,
                                  FullBinaryTreeParentNode<int, int> const &);
template std::ostream &operator<<(std::ostream &,
                                  FullBinaryTree<int, int> const &);

} // namespace FlexFlow
