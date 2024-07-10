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

} // namespace FlexFlow

#endif
