#include "utils/containers/get_all_assignments.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using K = value_type<0>;
using V = value_type<1>;

template
  std::unordered_set<std::unordered_map<K, V>> get_all_assignments(
      std::unordered_map<K, std::unordered_set<V>> const &);

} // namespace FlexFlow
