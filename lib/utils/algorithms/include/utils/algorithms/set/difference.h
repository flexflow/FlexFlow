#ifndef _FLEXFLOW_LIB_UTILS_ALGORITHMS_INCLUDE_UTILS_ALGORITHMS_SET_DIFFERENCE_H
#define _FLEXFLOW_LIB_UTILS_ALGORITHMS_INCLUDE_UTILS_ALGORITHMS_SET_DIFFERENCE_H

#include <unordered_set>

namespace FlexFlow {

template <typename T>
std::unordered_set<T> set_difference(std::unordered_set<T> const &l,
                                     std::unordered_set<T> const &r) {
  return filter(l, [&](T const &element) { return !contains(r, element); });
}

} // namespace FlexFlow

#endif
