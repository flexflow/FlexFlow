#ifndef _FLEXFLOW_LIB_UTILS_ALGORITHMS_INCLUDE_UTILS_ALGORITHMS_VECTOR_TRANSFORM_H
#define _FLEXFLOW_LIB_UTILS_ALGORITHMS_INCLUDE_UTILS_ALGORITHMS_VECTOR_TRANSFORM_H

namespace FlexFlow {

template <typename F, typename In, typename Out>
std::vector<Out> vector_transform(F const &f, std::vector<In> const &v) {
  return transform(v, f);
}

} // namespace FlexFlow

#endif
