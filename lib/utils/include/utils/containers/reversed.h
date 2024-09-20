#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_REVERSED_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_REVERSED_H

#include <vector>

namespace FlexFlow {

template <typename T>
std::vector<T> reversed(std::vector<T> const &t) {
  std::vector<T> result(std::crbegin(t), std::crend(t));
  return result;
}

} // namespace FlexFlow

#endif
