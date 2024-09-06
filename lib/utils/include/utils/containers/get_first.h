#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_GET_FIRST_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_GET_FIRST_H

#include <set>
#include <unordered_set>

namespace FlexFlow {

template <typename T>
T get_first(std::unordered_set<T> const &s) {
  return *s.cbegin();
}

template <typename T>
T get_first(std::set<T> const &s) {
  return *s.cbegin();
}

} // namespace FlexFlow

#endif
