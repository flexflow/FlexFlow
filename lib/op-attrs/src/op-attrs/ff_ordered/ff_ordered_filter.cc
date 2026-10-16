#include "op-attrs/ff_ordered/ff_ordered_filter.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using Elem = value_type<0>;
using F = std::function<bool(Elem const &)>;

template FFOrdered<Elem> ff_ordered_filter(FFOrdered<Elem> const &, F &&);

} // namespace FlexFlow
