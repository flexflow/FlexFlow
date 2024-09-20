#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_MAP_VALUES_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_MAP_VALUES_H

#include <type_traits>
#include <unordered_map>

namespace FlexFlow {

template <typename K,
          typename V,
          typename F,
          typename V2 = std::invoke_result_t<F, V>>
std::unordered_map<K, V2> map_values(std::unordered_map<K, V> const &m,
                                     F const &f) {
  std::unordered_map<K, V2> result;
  for (auto const &kv : m) {
    result.insert({kv.first, f(kv.second)});
  }
  return result;
}

} // namespace FlexFlow

#endif
