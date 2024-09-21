#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_DIM_ORDERED_FF_ORDERED_OF_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_DIM_ORDERED_FF_ORDERED_OF_H

#include "op-attrs/dim_ordered/dim_ordered.h"

namespace FlexFlow {

template <typename C, typename T = typename C::value_type>
FFOrdered<T> ff_ordered_of(C const &c) {
  return FFOrdered<T>{c.cbegin(), c.cend()};
}

} // namespace FlexFlow

#endif
