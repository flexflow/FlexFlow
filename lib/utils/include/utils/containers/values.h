#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_VALUES_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_VALUES_H

#include <unordered_set>

namespace FlexFlow {

template <typename C>
std::unordered_multiset<typename C::mapped_type> values(C const &c) {
  std::unordered_multiset<typename C::mapped_type> result;
  for (auto const &kv : c) {
    result.insert(kv.second);
  }
  return result;
}

} // namespace FlexFlow

#endif
