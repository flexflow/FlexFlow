#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_GROUP_BY_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_GROUP_BY_H

#include <type_traits>
#include <unordered_map>
#include <unordered_set>

namespace FlexFlow {

template <typename V, typename F, typename K = std::invoke_result_t<F, V>>
std::unordered_map<K, std::unordered_set<V>>
    group_by(std::unordered_set<V> const &vs, F f) {
  std::unordered_map<K, std::unordered_set<V>> result;
  for (V const &v : vs) {
    result[f(v)].insert(v);
  }
  return result;
}

} // namespace FlexFlow

#endif
