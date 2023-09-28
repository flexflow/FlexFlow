#ifndef _FLEXFLOW_LIB_UTILS_ALGORITHMS_INCLUDE_UTILS_ALGORITHMS_VECTOR_SPLIT_H
#define _FLEXFLOW_LIB_UTILS_ALGORITHMS_INCLUDE_UTILS_ALGORITHMS_VECTOR_SPLIT_H

#include <vector>

namespace FlexFlow {

template <typename T>
std::pair<std::vector<T>, std::vector<T>> split(std::vector<T> const &v,
                                                       std::size_t idx) {
  assert(v.size() > idx);

  std::vector<T> prefix(v.begin(), v.begin() + idx);
  std::vector<T> postfix(v.begin() + idx, v.end());
  return {prefix, postfix};
}


} // namespace FlexFlow

#endif
