#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_SET_MINUS_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_SET_MINUS_H

#include <set>
#include <unordered_set>

namespace FlexFlow {

template <typename T>
std::unordered_set<T> set_minus(std::unordered_set<T> const &l,
                                std::unordered_set<T> const &r) {
  std::unordered_set<T> result = l;
  for (T const &t : r) {
    result.erase(t);
  }
  return result;
}

template <typename T>
std::set<T> set_minus(std::set<T> const &l, std::set<T> const &r) {
  std::set<T> result = l;
  for (T const &t : r) {
    result.erase(t);
  }
  return result;
}

} // namespace FlexFlow

#endif
