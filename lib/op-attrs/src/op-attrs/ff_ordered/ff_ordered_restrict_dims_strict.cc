#include "op-attrs/ff_ordered/ff_ordered_restrict_dims_strict.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using T = value_type<0>;

template FFOrdered<T>
    ff_ordered_restrict_dims_strict(FFOrdered<T> const &,
                                    std::set<ff_dim_t> const &);

} // namespace FlexFlow
