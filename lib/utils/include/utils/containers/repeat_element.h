#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_REPLICATE_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_REPLICATE_H

#include "utils/nonnegative_int/nonnegative_int.h"
#include <fmt/format.h>
#include <vector>

namespace FlexFlow {

template <typename T>
std::vector<T> repeat_element(nonnegative_int num_times, T const &element) {
  std::vector<T> result;
  for (int i = 0; i < num_times; ++i) {
    result.push_back(element);
  }
  return result;
}

} // namespace FlexFlow

#endif
