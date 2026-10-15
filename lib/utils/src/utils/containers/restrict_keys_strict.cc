#include "utils/containers/restrict_keys_strict.h"
#include "utils/archetypes/ordered_value_type.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using K = ordered_value_type<0>;
using V = value_type<0>;

template std::map<K, V> restrict_keys_strict(std::map<K, V> const &,
                                             std::set<K> const &);

} // namespace FlexFlow
