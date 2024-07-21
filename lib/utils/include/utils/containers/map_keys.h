#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_MAP_KEYS_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_MAP_KEYS_H

#include <unordered_map>
#include <type_traits> 

namespace FlexFlow {

template <typename K,
          typename V,
          typename F,
          typename K2 = std::invoke_result_t<F, K>>
std::unordered_map<K2, V> map_keys(std::unordered_map<K, V> const &m,
                                   F const &f) {
  std::unordered_map<K2, V> result;
  for (auto const &kv : m) {
    result.insert({f(kv.first), kv.second});
  }
  return result;
}

} // namespace FlexFlow

#endif
