#ifndef _FLEXFLOW_UTILS_INCLUDE_UTILS_CONTAINERS_COMPARE_BY_H
#define _FLEXFLOW_UTILS_INCLUDE_UTILS_CONTAINERS_COMPARE_BY_H

#include <functional>

namespace FlexFlow {

template <typename T, typename F>
std::function<bool(T const &, T const &)> compare_by(F const &f) {
  return [=](T const &lhs, T const &rhs) { return f(lhs) < f(rhs); };
}

} // namespace FlexFlow

#endif
