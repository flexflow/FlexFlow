#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_REPLICATE_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_REPLICATE_H

#include <vector>

namespace FlexFlow {

template <typename T>
std::vector<T> replicate(int n, T const &element) {
  return std::vector<T>(n, element);
}

} // namespace FlexFlow

#endif
