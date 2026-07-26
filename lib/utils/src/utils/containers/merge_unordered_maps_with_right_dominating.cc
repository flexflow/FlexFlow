#include "utils/containers/merge_unordered_maps_with_right_dominating.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using K = value_type<0>;
using V = value_type<1>;

template std::unordered_map<K, V> merge_unordered_maps_with_right_dominating(
    std::vector<std::unordered_map<K, V>> const &);

} // namespace FlexFlow
