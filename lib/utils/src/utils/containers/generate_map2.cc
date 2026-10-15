#include "utils/containers/generate_map2.h"
#include "utils/archetypes/ordered_value_type.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using K = ordered_value_type<0>;
using V = value_type<1>;
using X = value_type<2>;

template std::map<K, V> generate_map2(std::vector<X> const &,
                                      std::function<K(X const &)> &&,
                                      std::function<V(X const &)> &&);

} // namespace FlexFlow
