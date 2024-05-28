#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_VECTOR_TRANSFORM_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_VECTOR_TRANSFORM_H

#include <vector>
#include <algorithm>

namespace FlexFlow {

template <typename F, typename In>
std::vector<std::invoke_result_t<F, In>> vector_transform(std::vector<In> const &v, F const &f) {
  using Out = std::invoke_result_t<F, In>;

  std::vector<Out> result;
  std::transform(v.cbegin(), v.cend(), std::back_inserter(result), f);
  return result;
}

} // namespace FlexFlow

#endif
