#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_UNIQUE_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_UNIQUE_H

#include "utils/containers/without_order.h"

namespace FlexFlow {

template <typename C, typename T = typename C::value_type>
std::unordered_set<T> unique(C const &c) {
  return without_order(c);
}

} // namespace FlexFlow

#endif
