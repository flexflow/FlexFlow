#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_REVERSED_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_REVERSED_H

namespace FlexFlow {

template <typename T>
T reversed(T const &t) {
  T r;
  for (auto i = t.cend() - 1; i >= t.begin(); i--) {
    r.push_back(*i);
  }
  return r;
}

} // namespace FlexFlow

#endif
