#include "utils/containers/require_three_keys.h"
#include "utils/archetypes/ordered_value_type.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using K = ordered_value_type<0>;
using V = value_type<1>;

template std::tuple<V, V, V>
    require_three_keys(std::map<K, V> const &, K const &, K const &, K const &);

} // namespace FlexFlow
