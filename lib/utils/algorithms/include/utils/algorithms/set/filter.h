#ifndef _FLEXFLOW_LIB_UTILS_ALGORITHMS_INCLUDE_UTILS_ALGORITHMS_SET_FILTER_H
#define _FLEXFLOW_LIB_UTILS_ALGORITHMS_INCLUDE_UTILS_ALGORITHMS_SET_FILTER_H

#include <unordered_set>

namespace FlexFlow {

template <typename T, typename F>
std::unordered_set<T> filter(std::unordered_set<T> const &v, F const &f) {
  std::unordered_set<T> result;
  for (T const &t : v) {
    if (f(t)) {
      result.insert(t);
    }
  }
  return result;
}


} // namespace FlexFlow

#endif
