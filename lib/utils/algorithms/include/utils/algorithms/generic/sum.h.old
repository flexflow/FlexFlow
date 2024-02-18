#ifndef _FLEXFLOW_LIB_UTILS_ALGORITHMS_INCLUDE_UTILS_ALGORITHMS_GENERIC_SUM_H
#define _FLEXFLOW_LIB_UTILS_ALGORITHMS_INCLUDE_UTILS_ALGORITHMS_GENERIC_SUM_H

#include "utils/algorithms/type/monoid/functions/msum.h"

namespace FlexFlow {

template <typename C>
auto sum(C const &c) {
  return msum(c);
}

template <typename C, typename F>
auto sum_where(C const &c, F const &f) {
  return msumWhere(c, f);
}

} // namespace FlexFlow

#endif
