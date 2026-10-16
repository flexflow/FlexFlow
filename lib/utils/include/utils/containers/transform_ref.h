#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_TRANSFORM_REF_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_TRANSFORM_REF_H

namespace FlexFlow {

template <typename T, typename F>
void transform_ref(T &r, F &&f) {
  r = f(r);
}

} // namespace FlexFlow

#endif
