#include "op-attrs/parallel_tensor_shape.h"
#include "utils/containers.h"
#include "utils/hash-utils.h"
#include "op-attrs/tensor_dims.h"

namespace FlexFlow {

int num_shard_dims(ParallelTensorShape const &s) {
  return num_shard_dims(s.dims);
}

std::unordered_set<ReplicaParallelDim> replica_dims(ParallelTensorShape const &s) {
  return replica_dims(s.dims);
}

int get_num_replicas(ParallelTensorShape const &shape) {
  return product(
      transform(replica_dims(shape),
                [](ReplicaParallelDim const &d) -> int { return d.degree; }));
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

ParallelTensorShape lift_to_parallel(TensorShape const &s) {
  return {lift_to_parallel(s.dims), s.data_type};
}

TensorShape get_tensor_shape_unsafe(ParallelTensorShape const &) {
  NOT_IMPLEMENTED();
}

TensorShape get_reduced_shape(ParallelTensorShape const &s) {
  return TensorShape{
    get_reduced_dims(s.dims),
    s.data_type,
  };
}

} // namespace FlexFlow
