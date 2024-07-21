#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_AS_VECTOR_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_AS_VECTOR_H

#include <vector>

namespace FlexFlow {

template <typename C, typename E = typename C::value_type>
std::vector<E> as_vector(C const &c) {
  std::vector<E> result(c.cbegin(), c.cend());
  return result;
}


} // namespace FlexFlow

#endif
