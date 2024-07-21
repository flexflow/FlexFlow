#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_RESTRICT_KEYS_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_RESTRICT_KEYS_H

#include <unordered_map>
#include <unordered_set>
#include "utils/containers/contains.h"

namespace FlexFlow {

template <typename K, typename V>
std::unordered_map<K, V> restrict_keys(std::unordered_map<K, V> const &m,
                                       std::unordered_set<K> const &mask) {
  std::unordered_map<K, V> result;
  for (auto const &kv : m) {
    if (contains(mask, kv.first)) {
      result.insert(kv);
    }
  }
  return result;
}

} // namespace FlexFlow

#endif
