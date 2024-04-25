#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_TENSOR_DIMS_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_TENSOR_DIMS_H

#include "op-attrs/parallel_tensor_dims_t.h"
#include "op-attrs/tensor_dims_t.h"

namespace FlexFlow {

FFOrdered<size_t> const &ff_ordered(TensorDims const &);

size_t dim_at_idx(TensorDims const &, ff_dim_t);

ParallelTensorDims lift_to_parallel(TensorDims const &);

} // namespace FlexFlow

#endif
