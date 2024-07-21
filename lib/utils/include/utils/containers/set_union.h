#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_SET_UNION_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_SET_UNION_H

#include <unordered_set>

namespace FlexFlow {

template <typename T>
std::unordered_set<T> set_union(std::unordered_set<T> const &l,
                                std::unordered_set<T> const &r) {
  std::unordered_set<T> result = l;
  result.insert(r.cbegin(), r.cend());
  return result;
}

template <typename C, typename T = typename C::value_type::value_type>
std::unordered_set<T> set_union(C const &sets) {
  std::unordered_set<T> result;
  for (std::unordered_set<T> const &s : sets) {
    for (T const &element : s) {
      result.insert(element);
    }
  }
  return result;
}


} // namespace FlexFlow

#endif
