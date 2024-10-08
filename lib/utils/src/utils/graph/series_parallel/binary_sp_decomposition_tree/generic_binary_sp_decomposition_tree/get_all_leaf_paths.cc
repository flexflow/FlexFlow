#include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/get_all_leaf_paths.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using Tree = value_type<0>;
using Series = value_type<1>;
using Parallel = value_type<2>;
using Leaf = value_type<3>;

template std::unordered_set<BinaryTreePath> get_all_leaf_paths(
    Tree const &tree,
    GenericBinarySPDecompositionTreeImplementation<Tree,
                                                   Series,
                                                   Parallel,
                                                   Leaf> const &);

} // namespace FlexFlow
