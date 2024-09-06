#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_ENUMERATE_VECTOR_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_ENUMERATE_VECTOR_H

#include <map>
#include <utility>
#include <vector>

namespace FlexFlow {

template <typename T>
std::map<int, T> enumerate_vector(std::vector<T> const &v) {
  std::map<int, T> result;
  for (int i = 0; i < v.size(); i++) {
    result.insert({i, v.at(i)});
  }
  return result;
}

} // namespace FlexFlow

#endif
