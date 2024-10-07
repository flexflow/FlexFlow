#include "utils/full_binary_tree/get_num_tree_nodes.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using Tree = value_type<0>;
using Parent = value_type<1>;
using Leaf = value_type<2>;

template 
  int get_num_tree_nodes(Tree const &,
                         FullBinaryTreeImplementation<Tree, Parent, Leaf> const &);

} // namespace FlexFlow
