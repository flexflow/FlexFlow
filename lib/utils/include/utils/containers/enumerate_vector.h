#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_ENUMERATE_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_ENUMERATE_H

#include <utility>
#include <vector>

namespace FlexFlow {

template <typename T>
std::vector<std::pair<int, T>> enumerate_vector(std::vector<T> const &v) {
  std::vector<std::pair<int, T>> result;
  for (int i = 0; i < v.size(); i++) {
    result.push_back({i, v.at(i)});
  }
  return result;
}

} // namespace FlexFlow

#endif
