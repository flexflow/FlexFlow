#include "op-attrs/parallel_tensor_dims.h"
#include "op-attrs/replica_parallel_dim.h"
#include "op-attrs/replica_parallel_dim_set.h"
#include "op-attrs/shard_parallel_dim.h"
#include "utils/containers.h"

namespace FlexFlow {

FFOrdered<ShardParallelDim> ff_ordered_shard_dims(ParallelTensorDims const &d) {
  return d.shard_dims;
}

std::unordered_set<ReplicaParallelDim>
    replica_dims(ParallelTensorDims const &d) {
  return get_replica_dims(d.replica_dims);
}

size_t num_shard_dims(ParallelTensorDims const &dims) {
  return dims.shard_dims.size();
}

int total_replica_degree(ParallelTensorDims const &dims) {
  return product(transform(replica_dims(dims), [](ReplicaParallelDim const &d) {
    return d.degree;
  }));
}

int total_shard_degree(ParallelTensorDims const &dims) {
  return product(transform(as_vector(dims.shard_dims),
                           [](ShardParallelDim const &d) { return d.degree; }));
}

int total_parallel_degree(ParallelTensorDims const &dims) {
  return total_replica_degree(dims) * total_shard_degree(dims);
}

bool is_valid(ParallelTensorDims const &dims) {
  return all_of(dims.shard_dims,
                [](ShardParallelDim const &d) { return is_valid(d); }) &&
         all_of(replica_dims(dims),
                [](ReplicaParallelDim const &d) { return is_valid(d); });
}

ShardParallelDim shard_dim_at_idx(ParallelTensorDims const &d, ff_dim_t idx) {
  if (idx.value < 0) {
    idx = ff_dim_t{d.shard_dims.size() + idx.value};
  }
  return d.shard_dims.at(idx);
}

ShardParallelDim &shard_dim_at_idx(ParallelTensorDims &d, ff_dim_t idx) {
  return d.shard_dims.at(idx);
}

TensorDims get_piece_dims(ParallelTensorDims const &) {
  NOT_IMPLEMENTED();
}

TensorDims get_tensor_dims_unsafe(ParallelTensorDims const &) {
  NOT_IMPLEMENTED();
}

TensorDims get_reduced_dims(ParallelTensorDims const &) {
  NOT_IMPLEMENTED();
}

} // namespace FlexFlow
