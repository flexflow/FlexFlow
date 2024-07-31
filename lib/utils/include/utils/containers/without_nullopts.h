#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_WITHOUT_NULLOPTS_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_WITHOUT_NULLOPTS_H

#include <optional>
#include <unordered_set>
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

template <typename T>
std::unordered_set<T>
    without_nullopts(std::unordered_set<std::optional<T>> const &s) {
  std::unordered_set<T> result;
  for (std::optional<T> const &t : s) {
    if (t.has_value()) {
      result.insert(t.value());
    }
  }
  return result;
}

} // namespace FlexFlow

#endif
