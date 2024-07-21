#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_INTERSECTION_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_INTERSECTION_H

#include <unordered_set>
#include "utils/containers/contains.h"
#include <optional>

namespace FlexFlow {

template <typename T>
std::unordered_set<T> intersection(std::unordered_set<T> const &l,
                                   std::unordered_set<T> const &r) {
  std::unordered_set<T> result;
  for (T const &ll : l) {
    if (contains(r, ll)) {
      result.insert(ll);
    }
  }
  return result;
}

template <typename C, typename T = typename C::value_type>
std::optional<T> intersection(C const &c) {
  std::optional<T> result;
  for (T const &t : c) {
    result = intersection(result.value_or(t), t);
  }

  return result;
}


} // namespace FlexFlow

#endif
