#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_FF_ORDERED_FF_ORDERED_FILTER_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_FF_ORDERED_FF_ORDERED_FILTER_H

#include "op-attrs/ff_ordered/ff_ordered.h"
#include "op-attrs/ff_ordered/ff_ordered_of.h"
#include "utils/containers/filter.h"
#include "utils/containers/vector_of.h"

namespace FlexFlow {

template <typename F, typename Elem>
FFOrdered<Elem> ff_ordered_filter(FFOrdered<Elem> const &v, F &&f) {
  return ff_ordered_of(filter(vector_of(v), f));
}

} // namespace FlexFlow

#endif
