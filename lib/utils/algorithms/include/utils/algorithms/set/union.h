#ifndef _FLEXFLOW_LIB_UTILS_ALGORITHMS_INCLUDE_UTILS_ALGORITHMS_SET_UNION_H
#define _FLEXFLOW_LIB_UTILS_ALGORITHMS_INCLUDE_UTILS_ALGORITHMS_SET_UNION_H

#include <unordered_set>

namespace FlexFlow {

template <typename T>
std::unordered_set<T> set_union(std::unordered_set<T> const &l,
                                std::unordered_set<T> const &r) {
  std::unordered_set<T> result = l;
  result.insert(r.cbegin(), r.cend());
  return result;
}


} // namespace FlexFlow

#endif
