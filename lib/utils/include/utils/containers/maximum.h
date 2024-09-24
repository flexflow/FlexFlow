#ifndef _FLEXFLOW_UTILS_INCLUDE_UTILS_CONTAINERS_MAXIMUM_H
#define _FLEXFLOW_UTILS_INCLUDE_UTILS_CONTAINERS_MAXIMUM_H

#include "utils/exception.h"
#include <algorithm>

namespace FlexFlow {

template <typename C>
typename C::value_type maximum(C const &c) {
  if (c.empty()) {
    throw mk_runtime_error(
        "input to function maximum must be non-empty container");
  }

  return *std::max_element(c.begin(), c.end());
}

} // namespace FlexFlow

#endif
