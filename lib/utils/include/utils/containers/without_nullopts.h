#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_WITHOUT_NULLOPTS_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_WITHOUT_NULLOPTS_H

#include <optional>
#include <vector>

namespace FlexFlow {

template <typename T>
std::vector<T> without_nullopts(std::vector<std::optional<T>> const &v) {
  std::vector<T> result;
  for (std::optional<T> const &t : v) {
    if (t.has_value()) {
      result.push_back(t.value());
    }
  }
  return result;
}

} // namespace FlexFlow

#endif
