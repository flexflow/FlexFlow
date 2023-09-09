#ifndef _FLEXFLOW_LIB_UTILS_ALGORITHMS_INCLUDE_UTILS_ALGORITHMS_EXTEND_H
#define _FLEXFLOW_LIB_UTILS_ALGORITHMS_INCLUDE_UTILS_ALGORITHMS_EXTEND_H

#include <vector>
#include <unordered_set>
#include <optional>

namespace FlexFlow {

template <typename T, typename C>
void extend(std::vector<T> &lhs, C const &rhs) {
  lhs.reserve(lhs.size() + std::distance(rhs.begin(), rhs.end()));
  lhs.insert(lhs.end(), rhs.begin(), rhs.end());
}

template <typename T, typename C>
void extend(std::unordered_set<T> &lhs, C const &rhs) {
  lhs.reserve(lhs.size() + std::distance(rhs.begin(), rhs.end()));
  lhs.insert(rhs.cbegin(), rhs.cend());
}

template <typename C, typename E>
void extend(C &lhs, std::optional<E> const &e) {
  if (e.has_value()) {
    return extend(lhs, e.value());
  }
}


} // namespace FlexFlow

#endif
