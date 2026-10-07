#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_TRY_MERGE_NONDISJOINT_MAPS_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_TRY_MERGE_NONDISJOINT_MAPS_H

#include <map>
#include <optional>

namespace FlexFlow {

template <typename K, typename V>
std::optional<std::map<K, V>>
    try_merge_nondisjoint_maps(std::map<K, V> m1, std::map<K, V> const &m2) {
  auto it = m1.begin();
  for (auto const &p : m2) {
    it = m1.emplace_hint(it, p);

    if (it->second != p.second) {
      return std::nullopt;
    }
  }

  return m1;
}

} // namespace FlexFlow

#endif
