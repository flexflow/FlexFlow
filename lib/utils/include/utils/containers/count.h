#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_COUNT_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_COUNT_H

#include <vector>
#include <cstddef>

namespace FlexFlow {

template <typename C, typename F>
int count(C const &c, F const &f) {
  int result = 0;
  for (auto const &v : c) {
    if (f(v)) {
      result++;
    }
  }
  return result;
}

std::vector<size_t> count(size_t n);

} // namespace FlexFlow

#endif
