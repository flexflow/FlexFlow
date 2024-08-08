#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_ALL_OF_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_ALL_OF_H

namespace FlexFlow {

template <typename C, typename F>
bool all_of(C const &c, F const &f) {
  for (auto const &v : c) {
    if (!f(v)) {
      return false;
    }
  }
  return true;
}

} // namespace FlexFlow

#endif
