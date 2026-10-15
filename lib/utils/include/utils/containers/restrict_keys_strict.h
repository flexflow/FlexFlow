#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_RESTRICT_KEYS_STRICT_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_RESTRICT_KEYS_STRICT_H

#include "utils/containers/is_subseteq_of.h"
#include "utils/containers/keys.h"
#include <libassert/assert.hpp>
#include <map>
#include <set>

namespace FlexFlow {

template <typename K, typename V>
std::map<K, V> restrict_keys_strict(std::map<K, V> const &m,
                                    std::set<K> const &mask) {
  ASSERT(is_subseteq_of(mask, keys(m)));

  std::map<K, V> result;
  for (auto const &kv : m) {
    if (contains(mask, kv.first)) {
      result.insert(kv);
    }
  }
  return result;
}

} // namespace FlexFlow

#endif
