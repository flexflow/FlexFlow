#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_SET_OF_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_SET_OF_H

#include <set>

namespace FlexFlow {

template <typename C, typename T = typename C::value_type>
std::set<T> set_of(C const &c) {
  std::set<T> result;
  for (T const &t : c) {
    result.insert(t);
  }
  return result;
}

} // namespace FlexFlow

#endif
