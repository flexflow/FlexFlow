#ifndef _FLEXFLOW_OP_ATTRS_INCLUDE_OP_ATTRS_PARALLEL_TENSOR_DIMS_H
#define _FLEXFLOW_OP_ATTRS_INCLUDE_OP_ATTRS_PARALLEL_TENSOR_DIMS_H

#include "op-attrs/parallel_dim.h"
#include "op-attrs/parallel_tensor_dim_degrees.dtg.h"
#include "op-attrs/parallel_tensor_dims.dtg.h"
#include "op-attrs/tensor_dims.dtg.h"

namespace FlexFlow {

FFOrdered<ShardParallelDim> ff_ordered_shard_dims(ParallelTensorDims const &);
FFOrdered<int> ff_ordered_shard_degrees(ParallelTensorDims const &);
std::unordered_set<ReplicaParallelDim> replica_dims(ParallelTensorDims const &);

/* size_t get_volume(ParallelTensorDims const &); */
size_t num_shard_dims(ParallelTensorDims const &);

ParallelTensorDimDegrees get_parallel_degrees(ParallelTensorDims const &);

ParallelTensorDims lift_to_parallel(TensorDims const &);
ParallelTensorDims
    lift_to_parallel_with_degrees(TensorDims const &,
                                  SumDegree const &,
                                  DiscardCopyDegree const &,
                                  FFOrdered<int> const &shard_degrees);
ParallelTensorDims
    lift_to_parallel_with_degrees(TensorDims const &,
                                  ParallelTensorDimDegrees const &);

int total_replica_degree(ParallelTensorDims const &);
int total_shard_degree(ParallelTensorDims const &);
int total_parallel_degree(ParallelTensorDims const &);

ShardParallelDim shard_dim_at_idx(ParallelTensorDims const &, ff_dim_t);
ShardParallelDim &shard_dim_at_idx(ParallelTensorDims &, ff_dim_t);

bool is_valid(ParallelTensorDims const &);
TensorDims get_piece_dims(ParallelTensorDims const &);
TensorDims get_tensor_dims_unsafe(ParallelTensorDims const &);

TensorDims get_reduced_dims(ParallelTensorDims const &);

} // namespace FlexFlow

#endif
