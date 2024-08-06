#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_RANGE_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_RANGE_H

#include <cassert>
#include <stdexcept>
#include <vector>

namespace FlexFlow {

template <typename T>
std::vector<T> range(T start, T end, T step = 1) {
  assert(step != 0);

  std::vector<T> result;
  if (step > 0) {
    for (T i = start; i < end; i += step) {
      result.push_back(i);
    }
  } else {
    for (T i = start; i > end; i += step) {
      result.push_back(i);
    }
  }
  return result;
}

template <typename T>
std::vector<T> range(T end) {
  return range(T(0), end);
}

} // namespace FlexFlow

#endif
