#include "utils/containers/unordered_restrict_keys_strict.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using K = value_type<0>;
using V = value_type<1>;

template std::unordered_map<K, V>
    unordered_restrict_keys_strict(std::unordered_map<K, V> const &,
                                   std::unordered_set<K> const &);

} // namespace FlexFlow
