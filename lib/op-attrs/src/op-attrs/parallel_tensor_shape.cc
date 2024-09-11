#include "op-attrs/parallel_tensor_shape.h"
#include "op-attrs/parallel_tensor_dims.h"
#include "op-attrs/tensor_dims.h"
#include "utils/containers/product.h"
#include "utils/containers/transform.h"
#include "utils/hash-utils.h"

namespace FlexFlow {

int num_shard_dims(ParallelTensorShape const &s) {
  return num_shard_dims(s.dims);
}

std::unordered_set<ReplicaParallelDim>
    replica_dims(ParallelTensorShape const &s) {
  return replica_dims(s.dims);
}

int get_num_replicas(ParallelTensorShape const &shape) {
  return product(
      transform(replica_dims(shape),
                [](ReplicaParallelDim const &d) -> int { return d.degree; }));
}

int get_sum_degree(ParallelTensorShape const &shape) {
  return shape.dims.replica_dims.sum_degree.value;
}

int get_discard_copy_degree(ParallelTensorShape const &shape) {
  return shape.dims.replica_dims.discard_copy_degree.value;
}

int get_total_parallel_degree(ParallelTensorShape const &s) {
  return total_parallel_degree(s.dims);
}

bool is_valid(ParallelTensorShape const &shape) {
  return is_valid(shape.dims);
}

ShardParallelDim shard_dim_at_idx(ParallelTensorShape const &s, ff_dim_t d) {
  return shard_dim_at_idx(s.dims, d);
}

ShardParallelDim &shard_dim_at_idx(ParallelTensorShape &s, ff_dim_t d) {
  return shard_dim_at_idx(s.dims, d);
}

FFOrdered<int> ff_ordered_shard_degrees(ParallelTensorShape const &s) {
  return ff_ordered_shard_degrees(s.dims);
}

std::optional<ShardParallelDim>
    try_get_shard_dim_at_idx(ParallelTensorShape const &s, ff_dim_t d) {
  if (s.dims.shard_dims.idx_is_valid(d)) {
    return s.dims.shard_dims.at(d);
  } else {
    return std::nullopt;
  }
}

ParallelTensorDimDegrees get_parallel_degrees(ParallelTensorShape const &s) {
  return get_parallel_degrees(s.dims); 
}

ParallelTensorShape lift_to_parallel(TensorShape const &s) {
  return ParallelTensorShape{lift_to_parallel(s.dims), s.data_type};
}

ParallelTensorShape
    lift_to_parallel_with_degrees(TensorShape const &s,
                                  SumDegree sum_degree,
                                  DiscardCopyDegree discard_copy_degree,
                                  FFOrdered<int> const &shard_degrees) {
  return ParallelTensorShape{
      lift_to_parallel_with_degrees(
          s.dims, sum_degree, discard_copy_degree, shard_degrees),
      s.data_type,
  };
}

ParallelTensorShape
    lift_to_parallel_with_degrees(TensorShape const &s, 
                                  ParallelTensorDimDegrees const &degrees) {
  return lift_to_parallel_with_degrees(s, degrees.sum_degree, degrees.discard_copy_degree, degrees.shard_degrees);
}

TensorShape require_not_parallel(ParallelTensorShape const &s) {
  int total_degree = get_total_parallel_degree(s);
  if (total_degree != 1) {
    throw mk_runtime_error(
        fmt::format("Error: require_not_parallel received a parallel tensor "
                    "shape with parallel degree {}: {}",
                    total_degree,
                    s));
  }

  return get_reduced_shape(s);
}

TensorShape get_tensor_shape_unsafe(ParallelTensorShape const &) {
  NOT_IMPLEMENTED();
}

TensorShape get_piece_shape(ParallelTensorShape const &s) {
  return get_reduced_shape(s);
}

TensorShape get_reduced_shape(ParallelTensorShape const &s) {
  return TensorShape{
      get_reduced_dims(s.dims),
      s.data_type,
  };
}

} // namespace FlexFlow
