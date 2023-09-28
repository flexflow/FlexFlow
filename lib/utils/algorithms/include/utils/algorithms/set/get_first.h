#ifndef _FLEXFLOW_LIB_UTILS_ALGORITHMS_INCLUDE_UTILS_ALGORITHMS_SET_GET_FIRST_H
#define _FLEXFLOW_LIB_UTILS_ALGORITHMS_INCLUDE_UTILS_ALGORITHMS_SET_GET_FIRST_H

namespace FlexFlow {

template <typename T>
T get_first(std::unordered_set<T> const &s) {
  return *s.cbegin();
}

} // namespace FlexFlow

#endif
