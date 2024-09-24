#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_GET_ONE_OF_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_GET_ONE_OF_H

#include "utils/exception.h"
#include <unordered_set>

namespace FlexFlow {

template <typename T>
T get_one_of(std::unordered_set<T> const &s) {
  if (s.empty()) {
    throw mk_runtime_error("input to are_all_same must be non-empty container");
  }
  return *s.cbegin();
}

} // namespace FlexFlow

#endif
