#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_TRY_AT_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_TRY_AT_H

#include <unordered_map>
#include <map>
#include "utils/containers/contains_key.h"
#include <optional>

namespace FlexFlow {

template <typename K, typename V>
std::optional<V> try_at(std::unordered_map<K, V> const &m, K const &k) {
  if (contains_key(m, k)) {
    return m.at(k);
  } else {
    return std::nullopt;
  }
}

template <typename K, typename V>
std::optional<V> try_at(std::map<K, V> const &m, K const &k) {
  if (contains_key(m, k)) {
    return m.at(k);
  } else {
    return std::nullopt;
  }
}

} // namespace FlexFlow

#endif
