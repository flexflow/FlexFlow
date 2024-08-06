#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_REPLICATE_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_REPLICATE_H

#include <cassert>
#include <iterator>

namespace FlexFlow {

template <typename C, typename T, typename F>
T foldl(C const &c, T init, F func) {
  T result = init;
  for (auto const &elem : c) {
    result = func(result, elem);
  }
  return result;
}

template <typename C, typename F>
auto foldl1(C const &c, F func) -> typename C::value_type {
  auto it = c.begin();
  assert(it != c.cend());

  typename C::value_type init = *it;
  ++it;
  C remaining(it, c.end());
  return foldl(remaining, init, func);
}

} // namespace FlexFlow

#endif
