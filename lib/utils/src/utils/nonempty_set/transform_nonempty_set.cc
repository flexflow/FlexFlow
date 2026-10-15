#include "utils/nonempty_set/transform_nonempty_set.h"
#include "utils/archetypes/ordered_value_type.h"

namespace FlexFlow {

using In = ordered_value_type<0>;
using Out = ordered_value_type<1>;

using F = std::function<Out(In const &)>;

template nonempty_set<Out> transform_nonempty_set(nonempty_set<In> const &,
                                                  F const &);

} // namespace FlexFlow
