#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_ANY_OF_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_ANY_OF_H

namespace FlexFlow {

template <typename C, typename F>
bool any_of(C const &c, F const &f) {
  for (auto const &v : c) {
    if (f(v)) {
      return true;
    }
  }
  return false;
}

} // namespace FlexFlow

#endif
