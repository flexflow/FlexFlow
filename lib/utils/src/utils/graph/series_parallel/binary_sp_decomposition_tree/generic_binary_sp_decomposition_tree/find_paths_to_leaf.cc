#include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/find_paths_to_leaf.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using Tree = value_type<0>;
using Series = value_type<1>;
using Parallel = value_type<2>;
using Leaf = value_type<3>;

template std::unordered_set<BinaryTreePath> find_paths_to_leaf(
    Tree const &,
    GenericBinarySPDecompositionTreeImplementation<Tree,
                                                   Series,
                                                   Parallel,
                                                   Leaf> const &,
    Leaf const &);

} // namespace FlexFlow
