#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_WITHOUT_ORDER_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_WITHOUT_ORDER_H

#include <unordered_set>

namespace FlexFlow {

template <typename C, typename T = typename C::value_type>
std::unordered_set<T> without_order(C const &c) {
  return {c.cbegin(), c.cend()};
}

} // namespace FlexFlow

#endif
