#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_FOLDL1_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_FOLDL1_H

#include "utils/exception.h"
#include <vector>

namespace FlexFlow {

template <typename T, typename F>
T foldl1(std::vector<T> const &vec, F f) {
  if (vec.empty()) {
    throw mk_runtime_error(fmt::format(
        "foldl1 expected non-empty vector, but receieved empty vector"));
  }

  auto it = vec.cbegin();
  T result = *it;
  it++;

  for (; it != vec.cend(); it++) {
    result = f(result, *it);
  }

  return result;
}

} // namespace FlexFlow

#endif
