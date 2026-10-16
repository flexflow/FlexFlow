#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_FF_ORDERED_FF_ORDERED_RESTRICT_DIMS_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_FF_ORDERED_FF_ORDERED_RESTRICT_DIMS_H

#include "op-attrs/ff_ordered/ff_ordered.h"
#include "op-attrs/ff_ordered/ff_ordered_get_idxs.h"
#include "utils/containers/contains.h"
#include <set>

namespace FlexFlow {

template <typename T>
FFOrdered<T> ff_ordered_restrict_dims(FFOrdered<T> const &input,
                                      std::set<ff_dim_t> const &restrict_to) {
  std::vector<T> result;

  for (ff_dim_t const &dim_idx : ff_ordered_get_idxs(input)) {
    if (contains(restrict_to, dim_idx)) {
      result.push_back(input.at(dim_idx));
    }
  }

  return FFOrdered<T>{result};
}

} // namespace FlexFlow

#endif
