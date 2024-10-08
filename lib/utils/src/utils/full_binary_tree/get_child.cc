#include "utils/full_binary_tree/get_child.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using Tree = value_type<0>;
using Parent = value_type<1>;
using Leaf = value_type<2>;

template Tree
    get_child(Parent const &,
              FullBinaryTreeImplementation<Tree, Parent, Leaf> const &,
              BinaryTreePathEntry const &);

} // namespace FlexFlow
