#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_FILTERMAP_VALUES_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_FILTERMAP_VALUES_H

#include <map>
#include <optional>
#include <type_traits>
#include <unordered_map>

namespace FlexFlow {

template <typename K,
          typename V,
          typename F,
          typename V2 = typename std::invoke_result_t<F, V>::value_type>
std::unordered_map<K, V2> filtermap_values(std::unordered_map<K, V> const &m,
                                           F const &f) {
  std::unordered_map<K, V2> result;
  for (auto const &[k, v] : m) {
    std::optional<V2> new_v = f(v);
    if (new_v.has_value()) {
      result.insert({k, new_v.value()});
    }
  }
  return result;
}

template <typename K,
          typename V,
          typename F,
          typename V2 = typename std::invoke_result_t<F, V>::value_type>
std::map<K, V2> filtermap_values(std::map<K, V> const &m, F const &f) {
  std::map<K, V2> result;
  for (auto const &[k, v] : m) {
    std::optional<V2> new_v = f(v);
    if (new_v.has_value()) {
      result.insert({k, new_v.value()});
    }
  }
  return result;
}

} // namespace FlexFlow

#endif
