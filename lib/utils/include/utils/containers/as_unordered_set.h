#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_AS_UNORDERED_SET_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_AS_UNORDERED_SET_H

#include <unordered_set>

namespace FlexFlow {

template <typename C, typename T = typename C::value_type>
std::unordered_set<T> as_unordered_set(C const &c) {
  return {c.cbegin(), c.cend()};
}

} // namespace FlexFlow

#endif
