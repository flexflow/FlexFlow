#ifndef _OP_META_PARALLEL_TENSOR_SHAPE_H
#define _OP_META_PARALLEL_TENSOR_SHAPE_H

#include "op-attrs/parallel_tensor_shape.dtg.h"
#include "op-attrs/replica_parallel_dim.dtg.h"
#include "op-attrs/tensor_shape.h"
#include <vector>

namespace FlexFlow {

int num_shard_dims(ParallelTensorShape const &);
ShardParallelDim shard_dim_at_idx(ParallelTensorShape const &, ff_dim_t);
ShardParallelDim &shard_dim_at_idx(ParallelTensorShape &, ff_dim_t);

FFOrdered<int> ff_ordered_shard_degrees(ParallelTensorShape const &);

std::optional<ShardParallelDim>
    try_get_shard_dim_at_idx(ParallelTensorShape const &, ff_dim_t);

ParallelTensorShape lift_to_parallel(TensorShape const &);
ParallelTensorShape
    lift_to_parallel_with_degrees(TensorShape const &,
                                  SumDegree sum_degree,
                                  DiscardCopyDegree discard_copy_degree,
                                  FFOrdered<int> const &shard_degrees);

std::unordered_set<ReplicaParallelDim>
    replica_dims(ParallelTensorShape const &);
TensorShape get_piece_shape(ParallelTensorShape const &);
int get_num_replica_dims(ParallelTensorShape const &);
int get_num_replicas(ParallelTensorShape const &);

int get_sum_degree(ParallelTensorShape const &);
int get_discard_copy_degree(ParallelTensorShape const &);

int get_total_parallel_degree(ParallelTensorShape const &);

bool is_valid(ParallelTensorShape const &);

TensorShape require_not_parallel(ParallelTensorShape const &);
TensorShape get_tensor_shape_unsafe(ParallelTensorShape const &);
std::vector<TensorShape>
    get_tensor_shapes_unsafe(std::vector<ParallelTensorShape> const &);

TensorShape get_reduced_shape(ParallelTensorShape const &);

} // namespace FlexFlow

#endif
