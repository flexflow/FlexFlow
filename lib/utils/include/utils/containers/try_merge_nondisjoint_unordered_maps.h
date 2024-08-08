#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_TRY_MERGE_NONDISJOINT_UNORDERED_MAPS_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_TRY_MERGE_NONDISJOINT_UNORDERED_MAPS_H

#include "utils/containers/contains_key.h"
#include <optional>
#include <unordered_map>

namespace FlexFlow {

template <typename K, typename V>
std::optional<std::unordered_map<K, V>>
    try_merge_nondisjoint_unordered_maps(std::unordered_map<K, V> const &m1,
                                         std::unordered_map<K, V> const &m2) {
  std::unordered_map<K, V> result;
  auto try_insert = [&](K const &k, V const &v) {
    if (contains_key(result, k) && result.at(k) != v) {
      return false;
    }
    result.insert({k, v});
    return true;
  };

  for (auto const &[k, v] : m1) {
    if (!try_insert(k, v)) {
      return std::nullopt;
    }
  }

  for (auto const &[k, v] : m2) {
    if (!try_insert(k, v)) {
      return std::nullopt;
    }
  }

  return result;
}

} // namespace FlexFlow

#endif
