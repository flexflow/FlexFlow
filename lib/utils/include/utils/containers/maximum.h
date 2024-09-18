#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_MAXIMUM_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_MAXIMUM_H

#include <algorithm>
#include <optional>

namespace FlexFlow {

template <typename C, typename E = typename C::value_type>
std::optional<E> maximum(C const &v) {
  if (v.empty()) {
    return std::nullopt;
  }

  return *std::max_element(std::cbegin(v), std::cend(v));
}

} // namespace FlexFlow

#endif
