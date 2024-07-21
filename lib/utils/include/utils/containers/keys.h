#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_KEYS_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_KEYS_H

#include <unordered_set>

namespace FlexFlow {

template <typename C>
std::unordered_set<typename C::key_type> keys(C const &c) {
  std::unordered_set<typename C::key_type> result;
  for (auto const &kv : c) {
    result.insert(kv.first);
  }
  return result;
}


} // namespace FlexFlow

#endif
