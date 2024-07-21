#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_ENUMERATE_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_ENUMERATE_H

#include "utils/bidict/bidict.h"
#include "utils/containers/enumerate_vector.h"
#include <unordered_set>

namespace FlexFlow {

template <typename T>
bidict<size_t, T> enumerate(std::vector<T> const &c) {
  return enumerate_vector(c);
}

template <typename T>
bidict<size_t, T> enumerate(std::unordered_set<T> const &c) {
  bidict<size_t, T> m;
  size_t idx = 0;
  for (auto const &v : c) {
    m.equate(idx++, v);
  }
  return m;
}


} // namespace FlexFlow

#endif
