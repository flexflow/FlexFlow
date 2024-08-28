#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_UNORDERED_MULTISET_OF_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_UNORDERED_MULTISET_OF_H

#include <unordered_set>

namespace FlexFlow {

template <typename C, typename T = typename C::value_type>
std::unordered_multiset<T> unordered_multiset_of(C const &c) {
  return {c.cbegin(), c.cend()};
}

} // namespace FlexFlow

#endif
