#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_GET_ONE_OF_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_GET_ONE_OF_H

#include "utils/exception.h"
#include "utils/fmt/unordered_set.h"
#include <unordered_set>
namespace FlexFlow {

template <typename T>
T get_one_of(std::unordered_set<T> const &s) {
  if (s.empty()) {
    throw mk_runtime_error(fmt::format(
        "get_one_of expected non-empty container but receieved {}", s));
  }
  return *s.cbegin();
}

} // namespace FlexFlow

#endif
