#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_GET_ONE_OF_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_GET_ONE_OF_H

#include <unordered_set>

namespace FlexFlow {

template <typename T>
T get_one_of(std::unordered_set<T> const &s) {
  return *s.cbegin();
}

} // namespace FlexFlow

#endif
