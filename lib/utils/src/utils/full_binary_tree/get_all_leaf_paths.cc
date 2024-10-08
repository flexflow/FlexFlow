#include "utils/full_binary_tree/get_all_leaf_paths.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

template std::unordered_set<BinaryTreePath>
    get_all_leaf_paths(value_type<0> const &,
                       FullBinaryTreeImplementation<value_type<0>,
                                                    value_type<1>,
                                                    value_type<2>> const &);

} // namespace FlexFlow
