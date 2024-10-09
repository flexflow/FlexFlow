#include "utils/containers/range.h"
#include <cassert>

namespace FlexFlow {

std::vector<int> range(int start, int end, int step) {
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
