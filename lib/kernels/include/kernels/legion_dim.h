#ifndef _FLEXFLOW_KERNELS_INCLUDE_KERNELS_LEGION_DIM_H
#define _FLEXFLOW_KERNELS_INCLUDE_KERNELS_LEGION_DIM_H

#include "op-attrs/dim_ordered.h"
#include "utils/strong_typedef.h"

namespace FlexFlow {

struct legion_dim_t : strong_typedef<legion_dim_t, int> {
  using strong_typedef::strong_typedef;
};

template <typename T>
using LegionOrdered = DimOrdered<legion_dim_t, T>;

using LegionTensorDims = LegionOrdered<size_t>;

} // namespace FlexFlow

#endif
