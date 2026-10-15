#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_REQUIRE_THREE_KEYS_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_REQUIRE_THREE_KEYS_H

#include "utils/containers/contains_key.h"
#include <libassert/assert.hpp>
#include <map>

namespace FlexFlow {

template <typename K, typename V>
std::tuple<V, V, V> require_three_keys(std::map<K, V> const &m,
                                       K const &k1,
                                       K const &k2,
                                       K const &k3) {
  ASSERT(k1 != k2);
  ASSERT(k1 != k3);
  ASSERT(k2 != k3);
  ASSERT(m.size() == 3);

  ASSERT(contains_key(m, k1));
  ASSERT(contains_key(m, k2));
  ASSERT(contains_key(m, k3));

  return {m.at(k1), m.at(k2), m.at(k3)};
}

} // namespace FlexFlow

#endif
