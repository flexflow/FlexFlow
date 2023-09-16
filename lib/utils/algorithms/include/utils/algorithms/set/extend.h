#ifndef _FLEXFLOW_LIB_UTILS_ALGORITHMS_INCLUDE_UTILS_ALGORITHMS_SET_EXTEND_H
#define _FLEXFLOW_LIB_UTILS_ALGORITHMS_INCLUDE_UTILS_ALGORITHMS_SET_EXTEND_H

#include <unordered_set>
#include "utils/type_traits_extra/is_optional.h"

namespace FlexFlow {

template <typename T, typename C, typename = std::enable_if_t<!is_optional_v<C>>>
void extend(std::unordered_set<T> &lhs, C const &rhs) {
  lhs.reserve(lhs.size() + std::distance(rhs.begin(), rhs.end()));
  lhs.insert(rhs.cbegin(), rhs.cend());
}

} // namespace FlexFlow

#endif
