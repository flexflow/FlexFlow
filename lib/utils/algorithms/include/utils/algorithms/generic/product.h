#ifndef _FLEXFLOW_LIB_UTILS_ALGORITHMS_INCLUDE_UTILS_ALGORITHMS_GENERIC_PRODUCT_H
#define _FLEXFLOW_LIB_UTILS_ALGORITHMS_INCLUDE_UTILS_ALGORITHMS_GENERIC_PRODUCT_H

#include "utils/algorithms/type/monoid/functions/mproduct.h"

namespace FlexFlow {

template <typename C>
auto product(C const &c) {
  return mproduct(c);
}

template <typename C, typename F>
auto product_where(C const &c, F const &f) {
  return mproductWhere(c, f);
}

} // namespace FlexFlow

#endif
