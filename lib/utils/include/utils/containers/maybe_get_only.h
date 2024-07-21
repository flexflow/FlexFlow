#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_MAYBE_GET_ONLY_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_MAYBE_GET_ONLY_H

#include <optional>

namespace FlexFlow {

template <typename C>
std::optional<typename C::value_type> maybe_get_only(C const &c) {
  if (c.size() == 1) {
    return *c.cbegin();
  } else {
    return std::nullopt;
  }
}

} // namespace FlexFlow

#endif
