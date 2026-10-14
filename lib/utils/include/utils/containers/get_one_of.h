#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_GET_ONE_OF_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_GET_ONE_OF_H

#include "utils/fmt/set.h"
#include <libassert/assert.hpp>
#include <set>

namespace FlexFlow {

template <typename T>
T get_one_of(std::set<T> const &s) {
  if (s.empty()) {
    PANIC(fmt::format(
        "get_one_of expected non-empty container but receieved {}", s));
  }
  return *s.cbegin();
}

} // namespace FlexFlow

#endif
