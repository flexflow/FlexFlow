#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_VALUES_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_VALUES_H

#include <vector>

namespace FlexFlow {

template <typename C>
std::vector<typename C::mapped_type> values(C const &c) {
  std::vector<typename C::mapped_type> result;
  for (auto const &kv : c) {
    result.push_back(kv.second);
  }
  return result;
}

} // namespace FlexFlow

#endif
