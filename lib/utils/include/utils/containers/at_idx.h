#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_AT_IDX_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_AT_IDX_H

#include <optional>
#include <vector>

namespace FlexFlow {

template <typename E>
std::optional<E> at_idx(std::vector<E> const &v, size_t idx) {
  if (idx >= v.size()) {
    return std::nullopt;
  } else {
    return v.at(idx);
  }
}

} // namespace FlexFlow

#endif
