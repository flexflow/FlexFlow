#ifndef _FLEXFLOW_LIB_UTILS_ALGORITHMS_INCLUDE_UTILS_ALGORITHMS_VECTOR_REPEAT_H
#define _FLEXFLOW_LIB_UTILS_ALGORITHMS_INCLUDE_UTILS_ALGORITHMS_VECTOR_REPEAT_H

#include <vector>

namespace FlexFlow {

template <typename F, typename Out>
std::vector<Out> repeat(int n, F const &f) {
  assert(n >= 0);

  std::vector<Out> result;
  for (int i = 0; i < n; i++) {
    result.push_back(f());
  }
  return result;
}

} // namespace FlexFlow

#endif
