#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_TRANSFORM_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_TRANSFORM_H

#include <type_traits>
#include <vector>
#include <algorithm>
#include "utils/containers/vector_transform.h"
#include "utils/required_core.h"

namespace FlexFlow {

template <typename F,
          typename In,
          typename Out = std::invoke_result_t<F, In>>
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

} // namespace FlexFlow

#endif
