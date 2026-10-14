#include "utils/many_to_one/many_to_one_transform_keys.h"
#include "utils/archetypes/ordered_value_type.h"

namespace FlexFlow {

using L1 = ordered_value_type<0>;
using L2 = ordered_value_type<1>;
using R = ordered_value_type<2>;
using F = std::function<L2(L1 const &)>;

template ManyToOne<L2, R> many_to_one_transform_keys(ManyToOne<L1, R> const &,
                                                     F &&);

} // namespace FlexFlow
