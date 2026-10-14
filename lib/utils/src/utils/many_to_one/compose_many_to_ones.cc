#include "utils/many_to_one/compose_many_to_ones.h"
#include "utils/archetypes/ordered_value_type.h"

namespace FlexFlow {

using T1 = ordered_value_type<0>;
using T2 = ordered_value_type<1>;
using T3 = ordered_value_type<2>;

template ManyToOne<T1, T3> compose_many_to_ones(ManyToOne<T1, T2> const &,
                                                ManyToOne<T2, T3> const &);

} // namespace FlexFlow
