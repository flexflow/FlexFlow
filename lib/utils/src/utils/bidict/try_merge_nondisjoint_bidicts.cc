#include "utils/bidict/try_merge_nondisjoint_bidicts.h"
#include "utils/archetypes/ordered_value_type.h"

namespace FlexFlow {

using L = ordered_value_type<0>;
using R = ordered_value_type<1>;

template std::optional<bidict<L, R>>
    try_merge_nondisjoint_bidicts(bidict<L, R> d1, bidict<L, R> const &d2);

} // namespace FlexFlow
