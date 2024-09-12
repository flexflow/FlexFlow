#ifndef _FLEXFLOW_UTILS_INCLUDE_UTILS_CONTAINERS_IS_SUBMAP_H
#define _FLEXFLOW_UTILS_INCLUDE_UTILS_CONTAINERS_IS_SUBMAP_H

#include "utils/containers/keys.h"
#include "utils/containers/restrict_keys.h"
#include <unordered_map>

namespace FlexFlow {

template <typename K, typename V>
bool is_submap(std::unordered_map<K, V> const &m,
               std::unordered_map<K, V> const &sub) {
  return restrict_keys(m, keys(sub)) == sub;
}

} // namespace FlexFlow

#endif
