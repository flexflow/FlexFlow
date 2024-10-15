#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_VECTOR_SPLIT_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_VECTOR_SPLIT_H

#include "utils/fmt/vector.h"
#include <cassert>
#include <stdexcept>
#include <vector>

namespace FlexFlow {

template <typename T>
std::pair<std::vector<T>, std::vector<T>> vector_split(std::vector<T> const &v,
                                                       int idx) {
  if (idx < 0 || idx > static_cast<int>(v.size())) {
    throw std::out_of_range(fmt::format(
        "Index out of range in vector_split: index = {}, vector = {}", idx, v));
  }

  std::vector<T> prefix(v.begin(), v.begin() + idx);
  std::vector<T> postfix(v.begin() + idx, v.end());
  return {prefix, postfix};
}

} // namespace FlexFlow

#endif
