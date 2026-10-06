#include "utils/graph/biindex.h"
#include "utils/archetypes/ordered_value_type.h"

namespace FlexFlow {

using K1 = ordered_value_type<0>;
using K2 = ordered_value_type<1>;
using Value = ordered_value_type<2>;

template class BiIndex<K1, K2, Value>;

} // namespace FlexFlow
