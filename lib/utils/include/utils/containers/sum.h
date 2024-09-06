#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_SUM_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_SUM_H

namespace FlexFlow {

template <typename C, typename T = typename C::value_type>
T sum(C const &c) {
  T result = 0;
  for (T const &t : c) {
    result += t;
  }
  return result;
}


} // namespace FlexFlow

#endif
