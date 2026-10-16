#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_FF_ORDERED_FF_ORDERED_REMOVE_SUFFIX_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_FF_ORDERED_FF_ORDERED_REMOVE_SUFFIX_H

#include "op-attrs/ff_ordered/ff_ordered.h"
#include "op-attrs/ff_ordered/ff_ordered_slice.h"
#include "utils/integer_conversions.h"
#include "utils/nonnegative_int/num_elements.h"

namespace FlexFlow {

template <typename T>
FFOrdered<T> ff_ordered_remove_suffix(FFOrdered<T> const &input,
                                      FFOrdered<T> const &suffix) {
  if (suffix.empty()) {
    return input;
  }

  relative_ff_dim_t first_suffix_dim_idx = relative_ff_dim_t{
      -1 * int_from_size_t(suffix.size()),
  };

  FFOrdered<T> input_suffix =
      ff_ordered_slice(input, first_suffix_dim_idx, std::nullopt);

  ASSERT(input_suffix == suffix);

  return ff_ordered_slice(input, relative_ff_dim_t{0}, first_suffix_dim_idx);
}

} // namespace FlexFlow

#endif
