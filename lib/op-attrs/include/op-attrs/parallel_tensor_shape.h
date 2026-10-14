#ifndef _OP_META_PARALLEL_TENSOR_SHAPE_H
#define _OP_META_PARALLEL_TENSOR_SHAPE_H

#include "op-attrs/ff_dim_t.h"
#include "op-attrs/num_ptensor_shard_dims_t.dtg.h"
#include "op-attrs/parallel_dim.h"
#include "op-attrs/parallel_tensor_dim_degrees.dtg.h"
#include "op-attrs/parallel_tensor_dim_idx_t.dtg.h"
#include "op-attrs/parallel_tensor_shape.dtg.h"
#include "op-attrs/replica_parallel_dim.dtg.h"
#include "op-attrs/tensor_shape.h"
#include <vector>

namespace FlexFlow {

num_ptensor_shard_dims_t num_shard_dims(ParallelTensorShape const &);
ShardParallelDim shard_dim_at_idx(ParallelTensorShape const &,
                                  relative_ff_dim_t);
ShardParallelDim &shard_dim_at_idx(ParallelTensorShape &, relative_ff_dim_t);

FFOrdered<positive_int> ff_ordered_shard_degrees(ParallelTensorShape const &);

std::optional<ShardParallelDim>
    try_get_shard_dim_at_idx(ParallelTensorShape const &, relative_ff_dim_t);

ParallelTensorDimDegrees get_parallel_degrees(ParallelTensorShape const &);

ParallelTensorShape lift_shape_to_parallel(TensorShape const &);
ParallelTensorShape lift_shape_to_parallel_with_degrees(
    TensorShape const &,
    SumDegree const &,
    DiscardCopyDegree const &,
    FFOrdered<positive_int> const &shard_degrees);
ParallelTensorShape
    lift_shape_to_parallel_with_degrees(TensorShape const &,
                                        ParallelTensorDimDegrees const &);

TensorShape get_piece_shape(ParallelTensorShape const &);
num_bytes_t get_piece_size_in_bytes(ParallelTensorShape const &);

std::set<ReplicaParallelDim> replica_dims(ParallelTensorShape const &);

positive_int get_num_replica_dims(ParallelTensorShape const &);
positive_int get_num_replicas(ParallelTensorShape const &);

positive_int get_sum_degree(ParallelTensorShape const &);
positive_int get_discard_copy_degree(ParallelTensorShape const &);

positive_int get_total_parallel_degree(ParallelTensorShape const &);

bool is_valid(ParallelTensorShape const &);

TensorShape require_not_parallel(ParallelTensorShape const &);
std::vector<TensorShape>
    get_tensor_shapes_unsafe(std::vector<ParallelTensorShape> const &);

TensorShape get_reduced_shape(ParallelTensorShape const &);

ParallelDim get_parallel_dim_at_idx(ParallelTensorShape const &shape,
                                    parallel_tensor_dim_idx_t idx);

std::set<parallel_tensor_dim_idx_t>
    get_parallel_tensor_dim_indices(ParallelTensorShape const &shape);

} // namespace FlexFlow

#endif
