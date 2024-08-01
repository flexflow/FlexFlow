#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_SET_DIFFERENCE_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_SET_DIFFERENCE_H

#include "utils/containers/contains.h"
#include "utils/containers/filter.h"
#include <unordered_set>

namespace FlexFlow {

template <typename T>
std::unordered_set<T> set_difference(std::unordered_set<T> const &l,
                                     std::unordered_set<T> const &r) {
  return filter(l, [&](T const &element) { return !contains(r, element); });
}

} // namespace FlexFlow

#endif
