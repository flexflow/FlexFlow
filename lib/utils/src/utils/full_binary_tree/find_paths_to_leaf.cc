#include "utils/full_binary_tree/find_paths_to_leaf.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using Tree = value_type<0>;
using Parent = value_type<1>;
using Leaf = value_type<2>;

template std::unordered_set<BinaryTreePath>
    find_paths_to_leaf(Tree const &,
                       FullBinaryTreeImplementation<Tree, Parent, Leaf> const &,
                       Leaf const &);

} // namespace FlexFlow
