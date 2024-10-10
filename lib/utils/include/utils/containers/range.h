#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_RANGE_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_RANGE_H

#include <vector>

namespace FlexFlow {

std::vector<int> range(int start, int end, int step = 1);
std::vector<int> range(int end);

} // namespace FlexFlow

#endif
