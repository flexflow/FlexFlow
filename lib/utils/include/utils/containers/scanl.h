#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_SCANL_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_SCANL_H

#include <vector>

namespace FlexFlow {

template <typename C, typename F, typename T>
std::vector<T> scanl(C const &c, T init, F const &op) {
  std::vector<T> result;

  result.push_back(init);

  for (auto const &elem : c) {
    init = op(init, elem);
    result.push_back(init);
  }

  return result;
}

template <typename C, typename F>
auto scanl1(C const &c, F op) {
  using T = typename C::value_type;

  if (c.empty()) {
    return std::vector<T>();
  }

  auto it = c.begin();
  T init = *it;
  ++it;

  C remaining(it, c.end());
  return scanl(remaining, init, op);
}

} // namespace FlexFlow

#endif
