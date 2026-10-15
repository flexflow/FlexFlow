#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_BINARY_MERGE_UNORDERED_MAPS_WITH_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_BINARY_MERGE_UNORDERED_MAPS_WITH_H

#include "utils/containers/generate_unordered_map.h"
#include "utils/containers/merge_unordered_maps_with_right_dominating.h"
#include "utils/containers/set_intersection.h"
#include "utils/containers/set_minus.h"
#include "utils/containers/unordered_keys.h"
#include "utils/containers/unordered_restrict_keys_strict.h"
#include <unordered_map>

namespace FlexFlow {

template <typename K, typename V, typename F>
std::unordered_map<K, V>
    binary_merge_unordered_maps_with(std::unordered_map<K, V> const &lhs,
                                     std::unordered_map<K, V> const &rhs,
                                     F &&f) {

  std::unordered_set<K> l_keys = unordered_keys(lhs);
  std::unordered_set<K> r_keys = unordered_keys(rhs);

  std::unordered_set<K> l_only_keys = set_minus(l_keys, r_keys);
  std::unordered_set<K> r_only_keys = set_minus(r_keys, l_keys);
  std::unordered_set<K> both_keys = set_intersection(r_keys, l_keys);

  std::unordered_map<K, V> l_only =
      unordered_restrict_keys_strict(lhs, l_only_keys);
  std::unordered_map<K, V> r_only =
      unordered_restrict_keys_strict(rhs, r_only_keys);

  std::unordered_map<K, V> merged = generate_unordered_map(
      both_keys, [&](K const &k) { return f(lhs.at(k), rhs.at(k)); });

  return merge_unordered_maps_with_right_dominating(std::vector{
      l_only,
      r_only,
      merged,
  });
}

} // namespace FlexFlow

#endif
