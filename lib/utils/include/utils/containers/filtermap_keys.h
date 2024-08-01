#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_FILTERMAP_KEYS_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_FILTERMAP_KEYS_H

#include <map>
#include <optional>
#include <type_traits>
#include <unordered_map>

namespace FlexFlow {

template <typename K,
          typename V,
          typename F,
          typename K2 = typename std::invoke_result_t<F, K>::value_type>
std::unordered_map<K2, V> filtermap_keys(std::unordered_map<K, V> const &m,
                                         F const &f) {
  std::unordered_map<K2, V> result;
  for (auto const &[k, v] : m) {
    std::optional<K2> new_k = f(k);
    if (new_k.has_value()) {
      result.insert({new_k.value(), v});
    }
  }
  return result;
}

template <typename K,
          typename V,
          typename F,
          typename K2 = typename std::invoke_result_t<F, K>::value_type>
std::map<K2, V> filtermap_keys(std::map<K, V> const &m, F const &f) {
  std::map<K2, V> result;
  for (auto const &[k, v] : m) {
    std::optional<K2> new_k = f(k);
    if (new_k.has_value()) {
      result.insert({new_k.value(), v});
    }
  }
  return result;
}

} // namespace FlexFlow

#endif
