#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_FF_ORDERED_FF_ORDERED_WITHOUT_DIMS_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_FF_ORDERED_FF_ORDERED_WITHOUT_DIMS_H

#include "op-attrs/ff_ordered/ff_ordered_get_idxs.h"
#include "op-attrs/ff_ordered/ff_ordered_restrict_dims_strict.h"
#include "utils/containers/set_minus.h"

namespace FlexFlow {

template <typename T>
FFOrdered<T> ff_ordered_without_dims(FFOrdered<T> const &input,
                                     std::set<ff_dim_t> const &without) {
  std::set<ff_dim_t> input_dims = ff_ordered_get_idxs(input);
  std::set<ff_dim_t> resulting_dims = set_minus(input_dims, without);

  return ff_ordered_restrict_dims_strict(input, resulting_dims);
}

} // namespace FlexFlow

#endif
