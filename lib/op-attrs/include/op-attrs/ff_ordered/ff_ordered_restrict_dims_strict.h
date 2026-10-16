#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_FF_ORDERED_FF_ORDERED_RESTRICT_DIMS_STRICT_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_FF_ORDERED_FF_ORDERED_RESTRICT_DIMS_STRICT_H

#include "op-attrs/ff_ordered/ff_ordered_get_idxs.h"
#include "op-attrs/ff_ordered/ff_ordered_restrict_dims.h"
#include "utils/containers/is_subseteq_of.h"
#include <libassert/assert.hpp>

namespace FlexFlow {

template <typename T>
FFOrdered<T>
    ff_ordered_restrict_dims_strict(FFOrdered<T> const &input,
                                    std::set<ff_dim_t> const &restrict_to) {
  std::set<ff_dim_t> input_dims = ff_ordered_get_idxs(input);
  ASSERT(is_subseteq_of(restrict_to, input_dims));

  return ff_ordered_restrict_dims(input, restrict_to);
}

} // namespace FlexFlow

#endif
