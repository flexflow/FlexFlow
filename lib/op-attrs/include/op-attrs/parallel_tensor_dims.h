#ifndef _FLEXFLOW_OP_ATTRS_INCLUDE_OP_ATTRS_PARALLEL_TENSOR_DIMS_H
#define _FLEXFLOW_OP_ATTRS_INCLUDE_OP_ATTRS_PARALLEL_TENSOR_DIMS_H

#include "op-attrs/parallel_dim.h"
#include "op-attrs/parallel_tensor_dims_t.h"
#include "op-attrs/tensor_dims_t.h"

namespace FlexFlow {

FFOrdered<ParallelDim> const &ff_ordered(ParallelTensorDims const &);

std::vector<ParallelDim> as_vector(ParallelTensorDims const &);

int get_num_replica_dims(ParallelTensorDims const &);

/* size_t get_volume(ParallelTensorDims const &); */
size_t num_dims(ParallelTensorDims const &);

ParallelDim dim_at_idx(ParallelTensorDims const &, ff_dim_t);
ParallelDim &dim_at_idx(ParallelTensorDims &, ff_dim_t);

bool is_valid(ParallelTensorDims const &);
TensorDims get_piece_dims(ParallelTensorDims const &);
TensorDims get_tensor_dims_unsafe(ParallelTensorDims const &);

} // namespace FlexFlow

#endif
