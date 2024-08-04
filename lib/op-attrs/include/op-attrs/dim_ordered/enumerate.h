#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_DIM_ORDERED_ENUMERATE_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_DIM_ORDERED_ENUMERATE_H

#include "op-attrs/dim_ordered.h"
#include "utils/bidict/bidict.h"

namespace FlexFlow {

template <typename T>
bidict<ff_dim_t, T> enumerate(FFOrdered<T> const &ff_ordered) {
  bidict<ff_dim_t, T> result;
  for (int raw_ff_dim : count(ff_ordered.size())) {
    ff_dim_t ff_dim = ff_dim_t{raw_ff_dim};
    result.equate({ff_dim, ff_ordered.at(ff_dim)});
  }
  return result;
}

} // namespace FlexFlow

#endif
