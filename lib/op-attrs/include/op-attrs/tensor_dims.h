#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_TENSOR_DIMS_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_TENSOR_DIMS_H

#include "op-attrs/parallel_tensor_dims.dtg.h"
#include "op-attrs/tensor_dims.dtg.h"

namespace FlexFlow {

FFOrdered<size_t> const &ff_ordered(TensorDims const &);

size_t num_dims(TensorDims const &);
size_t dim_at_idx(TensorDims const &, ff_dim_t);
size_t &dim_at_idx(TensorDims &, ff_dim_t);

bool tensor_dims_is_broadcastable_to(TensorDims const &curr,
                                     TensorDims const &goal);
std::optional<TensorDims>
    get_broadcast_target_dims(std::unordered_set<TensorDims> const &);

} // namespace FlexFlow

#endif
