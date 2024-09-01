#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_TRANSFORM_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_TRANSFORM_H

#include "utils/containers/vector_transform.h"
#include "utils/required_core.h"
#include <algorithm>
#include <type_traits>
#include <vector>
#include <optional>

namespace FlexFlow {

template <typename F, typename In, typename Out = std::invoke_result_t<F, In>>
std::vector<Out> transform(std::vector<In> const &v, F const &f) {
  return vector_transform(v, f);
}

template <typename F, typename C>
auto transform(req<C> const &c, F const &f)
    -> decltype(transform(std::declval<C>(), std::declval<F>())) {
  return transform(static_cast<C>(c), f);
}

template <typename F,
          typename In,
          typename Out = decltype(std::declval<F>()(std::declval<In>()))>
std::unordered_set<Out> transform(std::unordered_set<In> const &v, F const &f) {
  std::unordered_set<Out> result;
  for (auto const &e : v) {
    result.insert(f(e));
  }
  return result;
}

template <typename F>
std::string transform(std::string const &s, F const &f) {
  std::string result;
  std::transform(s.cbegin(), s.cend(), std::back_inserter(result), f);
  return result;
}

template <typename K,
          typename V,
          typename F,
          typename K2 = typename std::invoke_result_t<F, K, V>::first_type,
          typename V2 = typename std::invoke_result_t<F, K, V>::second_type>
std::unordered_map<K2, V2> transform(std::unordered_map<K, V> const &m,
                                     F const &f) {
  std::unordered_map<K2, V2> result;
  for (auto const &[k, v] : m) {
    result.insert(f(k, v));
  }
  return result;
}

template <typename F, typename T>
std::optional<std::invoke_result_t<F, T>> transform(std::optional<T> const &o,
                                                    F &&f) {
  using Return = std::invoke_result_t<F, T>;
  if (o.has_value()) {
    Return r = f(o.value());
    return std::optional<Return>{r};
  } else {
    return std::nullopt;
  }
}


} // namespace FlexFlow

#endif
