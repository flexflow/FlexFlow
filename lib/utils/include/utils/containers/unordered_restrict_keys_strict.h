#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_UNORDERED_RESTRICT_KEYS_STRICT_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_UNORDERED_RESTRICT_KEYS_STRICT_H

#include "utils/containers/is_subseteq_of.h"
#include "utils/containers/unordered_keys.h"
#include <libassert/assert.hpp>
#include <unordered_map>
#include <unordered_set>

namespace FlexFlow {

template <typename K, typename V>
std::unordered_map<K, V>
    unordered_restrict_keys_strict(std::unordered_map<K, V> const &m,
                                   std::unordered_set<K> const &mask) {
  ASSERT(is_subseteq_of(mask, unordered_keys(m)));

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
