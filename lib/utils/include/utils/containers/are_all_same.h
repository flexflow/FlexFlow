#ifndef _FLEXFLOW_UTILS_INCLUDE_UTILS_CONTAINERS_ARE_ALL_SAME_H
#define _FLEXFLOW_UTILS_INCLUDE_UTILS_CONTAINERS_ARE_ALL_SAME_H

#include "utils/exception.h"

namespace FlexFlow {

template <typename C>
bool are_all_same(C const &c) {
  if (c.empty()) {
    return true;
  }
  auto const &first = *c.cbegin();
  for (auto const &v : c) {
    if (v != first) {
      return false;
    }
  }
  return true;
}

} // namespace FlexFlow

#endif
