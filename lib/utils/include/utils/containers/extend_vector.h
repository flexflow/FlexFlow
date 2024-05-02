#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_EXTEND_VECTOR_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_EXTEND_VECTOR_H

#include <vector>

namespace FlexFlow {

template <typename T, typename C>
void extend_vector(std::vector<T> &lhs, C const &rhs) {
  lhs.reserve(lhs.size() + std::distance(rhs.begin(), rhs.end()));
  lhs.insert(lhs.end(), rhs.begin(), rhs.end());
}


} // namespace FlexFlow

#endif
