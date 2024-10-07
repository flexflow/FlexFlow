#include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/is_binary_sp_tree_right_associative.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using Tree = value_type<1>;
using Series = value_type<2>;
using Parallel = value_type<3>;
using Leaf = value_type<4>;

template 
  bool is_binary_sp_tree_right_associative(
      Tree const &tree,
      GenericBinarySPDecompositionTreeImplementation<Tree, Series, Parallel, Leaf> const &);

} // namespace FlexFlow
