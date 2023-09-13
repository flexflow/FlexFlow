#ifndef _FLEXFLOW_LIB_UTILS_ALGORITHMS_INCLUDE_UTILS_ALGORITHMS_EXTEND_H
#define _FLEXFLOW_LIB_UTILS_ALGORITHMS_INCLUDE_UTILS_ALGORITHMS_EXTEND_H

#include <vector>
#include <unordered_set>
#include <optional>
#include "utils/type_traits_extra/is_optional.h"

namespace FlexFlow {

template <typename T, typename C, typename = std::enable_if_t<!is_optional_v<C>>>
void extend(std::vector<T> &lhs, C const &rhs) {
  lhs.reserve(lhs.size() + std::distance(rhs.begin(), rhs.end()));
  lhs.insert(lhs.end(), rhs.begin(), rhs.end());
}

template <typename T, typename C, typename = std::enable_if_t<!is_optional_v<C>>>
void extend(std::unordered_set<T> &lhs, C const &rhs) {
  lhs.reserve(lhs.size() + std::distance(rhs.begin(), rhs.end()));
  lhs.insert(rhs.cbegin(), rhs.cend());
}

template <typename C>
void extend(C &lhs, std::optional<std::decay_t<typename C::value_type>> const &e) {
  if (e.has_value()) {
    extend(lhs, std::vector{e.value()});
  }
}

template <typename C>
void extend(C &lhs, std::nullopt_t) {
  return;
}

template <typename T1, typename T2>
T1 extended(T1 const &t1, T2 const &t2) {
  T1 result = t1;
  extend(result, t2);
  return result;
}


} // namespace FlexFlow

#endif
