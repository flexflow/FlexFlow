#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_SORTED_BY_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_SORTED_BY_H

#include "utils/containers/sorted.h"

namespace FlexFlow {

template <typename C, typename F, typename Elem = sort_value_type_t<C>>
std::vector<Elem> sorted_by(C const &c, F const &f) {
  std::vector<Elem> result(c.begin(), c.end());
  inplace_sorted_by(result, f);
  return result;
}

} // namespace FlexFlow

#endif
