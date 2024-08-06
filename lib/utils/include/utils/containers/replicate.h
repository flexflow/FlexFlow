#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_REPLICATE_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_REPLICATE_H

#include <unordered_set>

namespace FlexFlow {

template <typename T>
std::unordered_multiset<T> replicate(std::size_t n, T const &element) {
  std::unordered_multiset<T> result;
  for (std::size_t i = 0; i < n; ++i) {
    result.insert(element);
  }
  return result;
}

} // namespace FlexFlow

#endif
