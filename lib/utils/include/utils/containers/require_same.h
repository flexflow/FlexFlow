#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_REQUIRE_SAME_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_REQUIRE_SAME_H

#include "utils/exception.h"
#include <fmt/format.h>

namespace FlexFlow {

template <typename T>
T const &require_same(T const &l, T const &r) {
  if (l != r) {
    throw mk_runtime_error(
        fmt::format("require_same received non-equal inputs: {} != {}", l, r));
  }

  return l;
}

} // namespace FlexFlow

#endif
