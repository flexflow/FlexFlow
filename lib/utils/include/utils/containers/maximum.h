#ifndef _FLEXFLOW_UTILS_INCLUDE_UTILS_CONTAINERS_MAXIMUM_H
#define _FLEXFLOW_UTILS_INCLUDE_UTILS_CONTAINERS_MAXIMUM_H

#include "utils/exception.h"
#include "utils/fmt/map.h"
#include "utils/fmt/multiset.h"
#include "utils/fmt/set.h"
#include "utils/fmt/unordered_map.h"
#include "utils/fmt/unordered_set.h"
#include "utils/fmt/vector.h"
#include <algorithm>

namespace FlexFlow {

template <typename C>
typename C::value_type maximum(C const &c) {
  if (c.empty()) {
    throw mk_runtime_error(
        fmt::format("maximum expected non-empty container but received {}", c));
  }

  return *std::max_element(c.begin(), c.end());
}

} // namespace FlexFlow

#endif
