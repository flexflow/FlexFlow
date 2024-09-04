#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_FOLDR1_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_FOLDR1_H

#include <vector>
#include "utils/exception.h"

namespace FlexFlow {

template <typename T, typename F>
T foldr1(std::vector<T> const &vec, F f) {
  if (vec.empty()) {
    throw mk_runtime_error(fmt::format("foldr1 expected non-empty vector, but receieved empty vector"));
  }

  auto it = vec.crbegin();
  T result = *it;
  it++;
  for (; it != vec.crend(); it++) {
    result = f(result, *it);
  }

  return result;
}

} // namespace FlexFlow

#endif
