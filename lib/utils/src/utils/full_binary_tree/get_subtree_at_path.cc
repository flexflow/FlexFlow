#include "utils/full_binary_tree/get_subtree_at_path.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using Tree = value_type<0>;
using Parent = value_type<1>;
using Leaf = value_type<2>;

template std::optional<Tree> get_subtree_at_path(
    Tree const &,
    FullBinaryTreeImplementation<Tree, Parent, Leaf> const &,
    BinaryTreePath const &);

} // namespace FlexFlow
