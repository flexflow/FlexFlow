#ifndef _FLEXFLOW_UTILS_VECTOR_H
#define _FLEXFLOW_UTILS_VECTOR_H

#include <vector>

template <typename T>
std::vector<T> concat(std::vector<T> lhs, std::vector<T> const &rhs) {
  lhs.reserve(lhs.size() + rhs.size());
  lhs.insert(lhs.end(), rhs.cbegin(), rhs.cend());
  return lhs;
}

template <typename T, typename... Args>
std::vector<T> concat(std::vector<T> const &a, Args... rest) {
  return concat(a, concat(rest...));
}

#endif
