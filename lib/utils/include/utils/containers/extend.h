#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_EXTEND_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_EXTEND_H

#include "utils/containers/extend_vector.h"
#include <optional>
#include <unordered_set>

namespace FlexFlow {

template <typename T, typename C>
void extend(std::vector<T> &lhs, C const &rhs) {
  extend_vector(lhs, rhs);
}

template <typename T>
void extend(std::vector<T> &lhs, std::optional<T> const &rhs) {
  if (rhs.has_value()) {
    extend(lhs, std::vector<T>{rhs.value()});
  }
}

template <typename T, typename C>
void extend(std::unordered_set<T> &lhs, C const &rhs) {
  lhs.reserve(lhs.size() + std::distance(rhs.begin(), rhs.end()));
  lhs.insert(rhs.cbegin(), rhs.cend());
}

template <typename T>
void extend(std::unordered_set<T> &lhs, std::optional<T> const &rhs) {
  if (rhs.has_value()) {
    extend(lhs, std::vector<T>{rhs.value()});
  }
}

} // namespace FlexFlow

#endif
