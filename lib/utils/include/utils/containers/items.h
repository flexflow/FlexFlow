#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_ITEMS_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_ITEMS_H

#include <unordered_set>

namespace FlexFlow {

template <typename C>
std::unordered_set<std::pair<typename C::key_type, typename C::mapped_type>>
    items(C const &c) {
  return {c.begin(), c.end()};
}

} // namespace FlexFlow

#endif
