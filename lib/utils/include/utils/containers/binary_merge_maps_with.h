#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_BINARY_MERGE_MAPS_WITH_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_BINARY_MERGE_MAPS_WITH_H

#include "utils/containers/generate_map.h"
#include "utils/containers/keys.h"
#include "utils/containers/merge_maps_with_right_dominating.h"
#include "utils/containers/restrict_keys_strict.h"
#include "utils/containers/set_intersection.h"
#include "utils/containers/set_minus.h"
#include <map>

namespace FlexFlow {

template <typename K, typename V, typename F>
std::map<K, V> binary_merge_maps_with(std::map<K, V> const &lhs,
                                      std::map<K, V> const &rhs,
                                      F &&f) {

  std::set<K> l_keys = keys(lhs);
  std::set<K> r_keys = keys(rhs);

  std::set<K> l_only_keys = set_minus(l_keys, r_keys);
  std::set<K> r_only_keys = set_minus(r_keys, l_keys);
  std::set<K> both_keys = set_intersection(r_keys, l_keys);

  std::map<K, V> l_only = restrict_keys_strict(lhs, l_only_keys);
  std::map<K, V> r_only = restrict_keys_strict(rhs, r_only_keys);

  std::map<K, V> merged = generate_map(
      both_keys, [&](K const &k) { return f(lhs.at(k), rhs.at(k)); });

  return merge_maps_with_right_dominating(std::vector{
      l_only,
      r_only,
      merged,
  });
}

} // namespace FlexFlow

#endif
