#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_RANGE_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_RANGE_H

#include <cassert>
#include <stdexcept>
#include <vector>

namespace FlexFlow {

std::vector<int> range(int start, int end, int step = 1) {
  assert(step != 0);

  std::vector<int> result;
  if (step > 0) {
    for (int i = start; i < end; i += step) {
      result.push_back(i);
    }
  } else {
    for (int i = start; i > end; i += step) {
      result.push_back(i);
    }
  }
  return result;
}

std::vector<int> range(int end) {
  return range(0, end);
}

} // namespace FlexFlow

#endif
