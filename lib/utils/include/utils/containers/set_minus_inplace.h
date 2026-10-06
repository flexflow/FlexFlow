#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_SET_MINUS_INPLACE_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_SET_MINUS_INPLACE_H

#include <set>

namespace FlexFlow {

template <typename T>
void set_minus_inplace(std::set<T> &result, std::set<T> const &rhs) {
  for (T const &t : rhs) {
    result.erase(t);
  }
}

} // namespace FlexFlow

#endif
