#ifndef _FLEXFLOW_OP_ATTRS_INCLUDE_OP_ATTRS_PARALLEL_TENSOR_DIM_IDX_H
#define _FLEXFLOW_OP_ATTRS_INCLUDE_OP_ATTRS_PARALLEL_TENSOR_DIM_IDX_H

#include "op-attrs/parallel_tensor_dim_idx_t.dtg.h"
#include "op-attrs/parallel_tensor_shape.h"

namespace FlexFlow {

ParallelDim get_parallel_dim_at_idx(ParallelTensorShape const &shape,
                                    parallel_tensor_dim_idx_t idx);

std::unordered_set<parallel_tensor_dim_idx_t>
    get_parallel_tensor_indices(ParallelTensorShape const &shape);

} // namespace FlexFlow

#endif
