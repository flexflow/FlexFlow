#ifndef _FLEXFLOW_KERNELS_INCLUDE_KERNELS_LEGION_DIM_H
#define _FLEXFLOW_KERNELS_INCLUDE_KERNELS_LEGION_DIM_H

#include "kernels/legion_dim_t.dtg.h"
#include "op-attrs/dim_ordered.h"

namespace FlexFlow {

legion_dim_t add_to_legion_dim(legion_dim_t legion_dim, int value);

legion_dim_t legion_dim_from_ff_dim(ff_dim_t, int num_dimensions);

template <typename T>
using LegionOrdered = DimOrdered<legion_dim_t, T>;

using LegionTensorDims = LegionOrdered<size_t>;

template <typename T>
FFOrdered<T>
    ff_ordered_from_legion_ordered(LegionOrdered<T> const &legion_ordered) {
  return FFOrdered<T>(legion_ordered.rbegin(), legion_ordered.rend());
}

} // namespace FlexFlow

#endif
