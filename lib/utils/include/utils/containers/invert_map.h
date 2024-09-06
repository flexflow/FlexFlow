#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_INVERT_MAP_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_INVERT_MAP_H

#include <type_traits>
#include <unordered_map>
#include <unordered_set>

namespace FlexFlow {

template <typename K, typename V>
std::unordered_map<V, std::unordered_set<K>>
    invert_map(std::unordered_map<K, V> const &m) {
  std::unordered_map<V, std::unordered_set<K>> result;
  for (auto const &[key, value] : m) {
    result[value].insert(key);
  }
  return result;
}
} // namespace FlexFlow

#endif
