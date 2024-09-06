#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_MULTISET_UNION_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_MULTISET_UNION_H

#include <set>
#include <unordered_set>

namespace FlexFlow {

template <typename T>
std::unordered_multiset<T>
    multiset_union(std::unordered_multiset<T> const &lhs,
                   std::unordered_multiset<T> const &rhs) {
  std::unordered_multiset<T> result = lhs;

  for (T const &t : rhs) {
    result.insert(t);
  }

  return result;
}

template <typename T>
std::multiset<T> multiset_union(std::multiset<T> const &lhs,
                                std::multiset<T> const &rhs) {
  std::multiset<T> result = lhs;

  for (T const &t : rhs) {
    result.insert(t);
  }

  return result;
}

template <typename C, typename T = typename C::value_type::value_type>
std::unordered_multiset<T> multiset_union(C const &c) {
  std::unordered_multiset<T> result;
  for (auto const &s : c) {
    for (T const &element : s) {
      result.insert(element);
    }
  }

  return result;
}

} // namespace FlexFlow

#endif
