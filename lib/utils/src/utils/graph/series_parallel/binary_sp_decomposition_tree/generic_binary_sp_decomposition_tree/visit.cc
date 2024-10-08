#include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/visit.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using ReturnType = value_type<0>;
using Tree = value_type<1>;
using Series = value_type<2>;
using Parallel = value_type<3>;
using Leaf = value_type<4>;

template ReturnType
    visit(Tree const &,
          GenericBinarySPDecompositionTreeImplementation<Tree,
                                                         Series,
                                                         Parallel,
                                                         Leaf> const &,
          GenericBinarySPDecompositionTreeVisitor<ReturnType,
                                                  Tree,
                                                  Series,
                                                  Parallel,
                                                  Leaf> const &);

} // namespace FlexFlow
