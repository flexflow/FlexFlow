#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_IS_SUPERSETEQ_OF_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_IS_SUPERSETEQ_OF_H

#include "utils/containers/is_subseteq_of.h"
#include <unordered_set>

namespace FlexFlow {

template <typename T>
bool is_superseteq_of(std::unordered_set<T> const &super,
                      std::unordered_set<T> const &sub) {
  return is_subseteq_of<T>(sub, super);
}

} // namespace FlexFlow

#endif
