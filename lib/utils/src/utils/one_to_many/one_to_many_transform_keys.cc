#include "utils/one_to_many/one_to_many_transform_keys.h"
#include "utils/archetypes/ordered_value_type.h"

namespace FlexFlow {

using L1 = ordered_value_type<0>;
using L2 = ordered_value_type<1>;
using R = ordered_value_type<2>;
using F = std::function<L2(L1 const &)>;

template OneToMany<L2, R> one_to_many_transform_keys(OneToMany<L1, R> const &,
                                                     F &&f);

} // namespace FlexFlow
